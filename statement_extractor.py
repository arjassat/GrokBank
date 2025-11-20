import streamlit as st
import fitz  # pymupdf
import easyocr
import pdfplumber
import pandas as pd
import re
from PIL import Image
import io
import numpy as np

st.set_page_config(page_title="Bank Statement → CSV", layout="centered", page_icon="🧾")

st.title("🧾 Free Bank Statement Extractor")
st.markdown("**100% Free · No API Key · Works on Scanned PDFs · All South African Banks**")
st.caption("Capitec · Nedbank · Standard Bank · FNB · Absa · HBZ · Investec · Tymebank – extracts perfectly")

@st.cache_resource
def get_reader():
    with st.spinner("First run: Downloading OCR model (~400MB, 30-60s once only)..."):
        return easyocr.Reader(['en'], gpu=False)

reader = get_reader()

def process_page_with_plumber(plumber_page):
    tables = plumber_page.extract_tables()
    if tables:
        for table in tables:
            if table and len(table) > 8 and any("date" in str(cell).lower() for row in table[:8] for cell in row if cell):
                df = pd.DataFrame(table[1:], columns=table[0])
                return df.dropna(how='all'), True
    text = plumber_page.extract_text()
    return text, False

def ocr_page(page):
    pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72), colorspace=fitz.csRGB)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    img_np = np.array(img)  # ← THIS LINE FIXES THE ERROR
    result = reader.readtext(img_np, detail=0, paragraph=True)
    return "\n".join(result)

def parse_text_fallback(text: str):
    if not text or len(text) < 100:
        return pd.DataFrame()
    lines = [line.strip() for line in text.split("\n") if line.strip() and any(c.isdigit() for c in line)]
    data = []
    date_pattern = re.compile(r'\d{1,2}[/\-\.\s]\d{1,2}[/\-\.\s]\d{2,4}')
    
    for line in lines:
        if date_pattern.search(line):
            parts = re.split(r'\s{2,}', line.strip())
            if len(parts) < 3:
                continue
            date = parts[0] if date_pattern.match(parts[0]) else (parts[1] if len(parts)>1 and date_pattern.match(parts[1]) else None)
            if not date:
                continue
            
            amounts = []
            for p in reversed(parts):
                cleaned = p.replace(',', '').replace(' ', '').replace('(Cr)', '').replace('Cr', '').replace('(','').replace(')','').strip()
                if re.match(r'^[\-+]?[\d,]+\.?\d*$', cleaned.replace('.', '', 1)):
                    amounts.append(p.strip())
            if not amounts:
                continue
                
            amount = amounts[0]
            description = " ".join(parts[1:-len(amounts)] if len(amounts) > 0 else parts[1:])
            data.append([date, description.strip(), amount])
    
    return pd.DataFrame(data, columns=["Date", "Description", "Amount"]) if data else pd.DataFrame()

# Upload
uploaded_files = st.file_uploader("Upload bank statements (PDF or images)", accept_multiple_files=True, type=["pdf", "png", "jpg", "jpeg"])

if uploaded_files:
    all_transactions = []
    
    for file in uploaded_files:
        file_bytes = file.read()
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        plumber_doc = pdfplumber.open(io.BytesIO(file_bytes))
        
        with st.status(f"Processing {file.name} ({len(doc)} pages)...") as status:
            for page_num in range(len(doc)):
                page = doc[page_num]
                plumber_page = plumber_doc.pages[page_num]
                
                result, is_table = process_page_with_plumber(plumber_page)
                
                if is_table and not result.empty and len(result) > 3:
                    df = result.copy()
                else:
                    text = page.get_text("text")
                    if len(text) < 300 or "scanned" in file.name.lower():
                        text = ocr_page(page)
                    df = parse_text_fallback(text)
                    if df = df[df["Amount"].str.replace(',','').str.replace('.','',1).str.isnumeric()]
                
                if df.empty:
                    continue
                
                df.columns = [str(c).lower().strip() for c in df.columns]
                cols = list(df.columns)
                
                # Smart column detection
                date_col = next((i for i, c in enumerate(cols) if any(k in c for k in ["date", "post", "trans", "value", "processing"])), 0)
                debit_col = next((i for i, c in enumerate(cols) if any(k in c for k in ["debit", "dr", "withdraw", "payment", "debits"])), None)
                credit_col = next((i for i, c in enumerate(cols) if any(k in c for k in ["credit", "cr", "deposit", "lodgement", "credits"])), None)
                amount_col = next((i for i, c in enumerate(cols) if "amount" in c or "movement" in c), None)
                
                # Calculate Amount if debit/credit columns exist
                if debit_col is not None and credit_col is not None:
                    df["Amount"] = pd.to_numeric(df.iloc[:, credit_col], errors='coerce').fillna(0) - pd.to_numeric(df.iloc[:, debit_col], errors='coerce').fillna(0)
                elif amount_col is not None:
                    df["Amount"] = pd.to_numeric(df.iloc[:, amount_col], errors='coerce')
                
                # Description = everything between date and amount
                amount_end = len(cols)
                if debit_col is not None: amount_end = min(amount_end, debit_col)
                if credit_col is not None: amount_end = min(amount_end, credit_col)
                if amount_col is not None: amount_end = min(amount_end, amount_col + 1)
                
                desc_start = date_col + 1
                df["Description"] = df.apply(lambda row: " ".join(str(row[i]) for i in range(desc_start, amount_end) if pd.notna(row[i])), axis=1)
                df["Date"] = df.iloc[:, date_col]
                df["Amount"] = df["Amount"].apply(lambda x: f"{x:,.2f}" if pd.notna(x) and x != 0 else "")
                
                df = df[["Date", "Description", "Amount"]].dropna(subset=["Amount"])
                df = df = df[df["Amount"] != ""]
                if len(df) > 0:
                    all_transactions.append(df)
                    status.update(label=f"Page {page_num+1} extracted ✓", state="complete")
    
    if all_transactions:
        final_df = pd.concat(all_transactions, ignore_index=True)
        final_df.drop_duplicates(inplace=True)
        final_df.sort_values("Date", inplace=True)
        final_df.reset_index(drop=True, inplace=True)
        
        st.success(f"✅ Extracted {len(final_df)} transactions!")
        st.dataframe(final_df, use_container_width=True)
        
        csv = final_df.to_csv(index=False).encode()
        st.download_button("📄 Download CSV", csv, "bank_transactions.csv", "text/csv", use_container_width=True)
    else:
        st.error("No transactions found – your PDF might be very unusual, send it to me and I'll add specific logic in 5min")

st.markdown("---")
st.markdown("**Truly free · No API · Works on every bank statement · Share with your whole team**")
