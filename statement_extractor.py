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

# One-time setup for dependencies (runs on app start)
if not os.path.exists('/usr/bin/tesseract'):
    st.info("Installing required system dependencies...")
    os.system('apt-get update')
    os.system('apt-get install -y tesseract-ocr poppler-utils')
    st.info("Dependencies installed.")

st.title("Bank Statement Extractor")
st.markdown("""
This app extracts transactions from bank statement PDFs (text or scanned) using free AI.
It outputs a CSV with columns: date, description, amount.
Powered by pytesseract for OCR and Groq's free Llama model for extraction.
""")

api_key = st.text_input("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", type="password")
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
Extract all transactions from this bank statement text chunk. Each transaction should have date, description, amount (positive for credit, negative for debit if applicable).
Ignore non-transaction text like headers or footers.
Output ONLY a JSON list of objects like: [{{"date": "YYYY-MM-DD", "description": "text", "amount": "number"}}]
If no transactions, output empty list [].

Text: {chunk}
"""

            try:
                completion = client.chat.completions.create(
                    model="llama3-70b-8192",  # Free model on Groq
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=4096,
                )
                json_str = completion.choices[0].message.content.strip()
                if json_str.startswith('```json'):
                    json_str = json_str.split('```json')[1].split('```')[0].strip()
                transactions = json.loads(json_str)
                file_transactions.extend(transactions)
            except Exception as e:
                st.error(f"AI extraction failed for {file.name} chunk {chunk_idx+1}: {e}")
                continue

        all_transactions.extend(file_transactions)
        progress_bar.progress((idx + 1) / total_files)

    if all_transactions:
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
