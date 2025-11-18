# app.py
import streamlit as st
import pdfplumber
from pdf2image import convert_from_bytes
import pytesseract
from PIL import Image
import io
import pandas as pd
from groq import Groq
import json
import os

st.title("Bank Statement Extractor")
st.markdown("""
This app extracts transactions from bank statement PDFs (text or scanned) from various South African banks (e.g., Capitec, FNB, Standard Bank, Nedbank, HBZ) using free AI.
It handles different formats automatically.
It outputs a CSV with columns: date, description, amount.
Powered by pytesseract for OCR and Groq's free Llama model for extraction.
""")

api_key = st.text_input("Enter your Groq API Key (get free at console.groq.com)", type="password")
if not api_key:
    st.warning("Please enter your Groq API key to proceed.")
    st.stop()

uploaded_files = st.file_uploader("Upload PDF bank statements", type="pdf", accept_multiple_files=True)

if uploaded_files and st.button("Extract and Download CSV"):
    all_transactions = []
    progress_bar = st.progress(0)
    total_files = len(uploaded_files)

    client = Groq(api_key=api_key)

    for idx, file in enumerate(uploaded_files):
        st.write(f"Processing {file.name}...")
        text = ""

        # Try extracting text directly (for searchable PDFs)
        try:
            with pdfplumber.open(file) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except Exception as e:
            st.warning(f"Error extracting text from {file.name}: {e}")

        # If no text extracted, use OCR
        if not text.strip():
            st.write(f"Performing OCR on {file.name}...")
            try:
                images = convert_from_bytes(file.getvalue())
                for img in images:
                    text += pytesseract.image_to_string(img) + "\n"
            except Exception as e:
                st.error(f"OCR failed for {file.name}: {e}")
                continue

        if not text.strip():
            st.error(f"No text extracted from {file.name}. Skipping.")
            continue

        # Chunk text if too large (to avoid token limits, ~8000 chars per chunk)
        chunks = [text[i:i+8000] for i in range(0, len(text), 8000)]
        file_transactions = []

        for chunk_idx, chunk in enumerate(chunks):
            prompt = f"""
You are an expert at extracting transactions from South African bank statements in various formats (e.g., Capitec, FNB, Standard Bank, Nedbank, HBZ, etc.). 
The text may come from scanned or text PDFs and could be messy due to OCR errors. Ignore that and focus on identifying transaction data.

Key rules:
- Extract ALL transactions, including debits, credits, fees, interest, etc.
- Date: Standardize to YYYY-MM-DD. Infer year from context if missing (e.g., use statement date or current year). Handle formats like DD/MM/YY, DD MMM YYYY, etc.
- Description: Clean and concise text, combining reference/description fields if needed. Remove unnecessary details like card numbers or auth codes unless relevant.
- Amount: Numeric value. Use positive for credits/deposits, negative for debits/withdrawals/fees. Parse currencies (assume ZAR if not specified). Fix OCR errors (e.g., '1,000.00' or '1000,00').
- Ignore non-transaction text: headers, footers, summaries, balances, account details, charts, VAT totals, etc.
- Handle multi-page or tabular formats: Look for columns like Post Date, Trans Date, Description, Reference, Fees, Amount, Balance.
- If duplicate transactions across chunks, include only once (but since chunks are sequential, process as is).
- For banks like Capitec: Columns often Post Date, Trans Date, Description, Reference, Fees, Amount, Balance.
- For FNB: Date, Description, Amount, Balance, Accrued Charges.
- For Standard Bank: Details, Service Fee, Debits, Credits, Date, Balance.
- For Nedbank: Tran list no, Date, Description, Fees, Debits, Credits, Balance.
- For HBZ: Date, Particulars, Debit, Credit.
- Adapt to variations; the AI should generalize.

Always output valid JSON, even if empty []. Do not wrap in markdown or add any other text. Start directly with [ and end with ].
Output ONLY a JSON list of objects like: [{"date": "YYYY-MM-DD", "description": "text", "amount": "number"}]
Sort by date ascending. If no transactions, output empty list [].

Text: {chunk}
"""

            try:
                completion = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",  # Updated model
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=4096,
                )
                json_str = completion.choices[0].message.content.strip()
                if json_str.startswith('```json'):
                    json_str = json_str.split('```json')[1].split('```')[0].strip()
                if not json_str:
                    transactions = []
                else:
                    transactions = json.loads(json_str)
                file_transactions.extend(transactions)
            except json.JSONDecodeError as je:
                st.error(f"Invalid JSON response for {file.name} chunk {chunk_idx+1}: {json_str}")
                continue
            except Exception as e:
                st.error(f"AI extraction failed for {file.name} chunk {chunk_idx+1}: {e}")
                continue

        # Deduplicate and sort transactions across chunks/files if needed
        unique_transactions = {f"{t['date']}_{t['description']}_{t['amount']}": t for t in file_transactions if 'date' in t and 'description' in t and 'amount' in t}.values()
        file_transactions = sorted(list(unique_transactions), key=lambda x: x['date'])

        all_transactions.extend(file_transactions)
        progress_bar.progress((idx + 1) / total_files)

    if all_transactions:
        # Global dedup and sort
        unique_all = {f"{t['date']}_{t['description']}_{t['amount']}": t for t in all_transactions if 'date' in t and 'description' in t and 'amount' in t}.values()
        all_transactions = sorted(list(unique_all), key=lambda x: x['date'])

        df = pd.DataFrame(all_transactions)
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="extracted_transactions.csv",
            mime="text/csv"
        )
        st.success("Extraction complete!")
        st.dataframe(df)  # Preview
    else:
        st.warning("No transactions extracted.")
