import streamlit as st
from groq import Groq
import pdfplumber
import pytesseract
from PIL import Image
import io
import pandas as pd
import os

# --- Config ---
st.set_page_config(page_title="Free Bank Statement Extractor", layout="centered")
st.title("🧾 Free AI Bank Statement to CSV")
st.caption("Upload scanned or digital PDFs → Get perfect Date | Description | Amount CSV. 100% free & unlimited.")

# Groq is free as of 2025 (use your free key)
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]  # we'll set this in Streamlit secrets

client = Groq(api_key=GROQ_API_KEY)

def extract_with_ai(image):
    # Convert image to base64
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = buffered.getvalue().hex()

    try:
        chat_completion = client.chat.completions.create(
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Extract all transactions from this bank statement image exactly as: Date (YYYY-MM-DD or DD/MM/YYYY), full Description, Amount (with - for debits if present). Return ONLY a markdown table, no extra text."},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_str}"}}
                ]
            }],
            model="llama-3.2-11b-vision-preview",  # or llama-3.2-90b-vision-preview when available
            temperature=0.1,
            max_tokens=2000
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        st.error("AI temporarily busy, falling back to OCR...")
        return None

def ocr_fallback(image):
    text = pytesseract.image_to_string(image, lang='eng')
    # Very simple regex-based extraction as fallback
    lines = [l for l in text.split('\n') if any(c.isdigit() for c in l)]
    data = []
    for line in lines:
        parts = line.split()
        if len(parts) > 3 and any(p.replace('.', '').replace(',', '').isdigit() for p in parts[-3:]):
            amount = parts[-1]
            description = " ".join(parts[:-3]) if len(parts)>5 else " ".join(parts[:-2])
            date = parts[0] if '/' in parts[0] or '-' in parts[0] else parts[1]
            data.append([date, description, amount])
    return pd.DataFrame(data, columns=["Date", "Description", "Amount"]).to_markdown()

# Upload
uploaded_files = st.file_uploader("Upload bank statements (PDF or images)", accept_multiple_files=True, 
                                 type=["pdf", "png", "jpg", "jpeg"])

if uploaded_files:
    all_transactions = []
    
    for uploaded_file in uploaded_files:
        if uploaded_file.type == "application/pdf":
            with pdfplumber.open(uploaded_file) as pdf:
                for page in pdf.pages:
                    img = page.to_image(resolution=300).original
                    result = extract_with_ai(img)
                    if result and "date" in result.lower():
                        df = pd.read_markdown(result, delim="|").dropna()
                    else:
                        markdown = ocr_fallback(img)
                        df = pd.read_markdown(markdown, delim="|").dropna()
                    all_transactions.append(df)
        else:
            image = Image.open(uploaded_file)
            result = extract_with_ai(image)
            if result and "date" in result.lower():
                df = pd.read_markdown(result, delim="|").dropna()
            else:
                markdown = ocr_fallback(image)
                df = pd.read_markdown(markdown, delim="|").dropna()
            all_transactions.append(df)
    
    if all_transactions:
        final_df = pd.concat(all_transactions, ignore_index=True)
        final_df = final_df.loc[:, ~final_df.columns.str.contains("^Unnamed")]  # clean
        st.success(f"Extracted {len(final_df)} transactions!")
        st.dataframe(final_df)
        
        csv = final_df.to_csv(index=False).encode()
        st.download_button("📄 Download CSV", csv, "bank_transactions.csv", "text/csv")

st.markdown("---")
st.markdown("Built 100% free • Uses Groq + Llama 3.2 Vision • Share this link with anyone!")
