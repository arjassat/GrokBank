import streamlit as st
import fitz  # pymupdf
import easyocr
import pdfplumber
import pandas as pd
import re
from PIL import Image
import io

st.set_page_config(page_title="Bank Statement → CSV (100% Free)", layout="centered", page_icon="🧾")

st.title("🧾 Bank Statement to CSV Extractor")
st.markdown("**100% Free · No API Key · Works on Scanned PDFs · Capitec, Nedbank, Standard Bank, FNB, HBZ, etc.**")
st.caption("Upload one or many PDFs (even scanned) → Get clean Date · Description · Amount CSV. Share the link with coworkers — no login needed.")

@st.cache_resource
def get_reader():
    return easyocr.Reader(['en'], gpu=False)  # downloads model on first run (~40s)

reader = get_reader()

def process_page_with_plumber(page):
    tables = page.extract_tables()
    if tables:
        # Find biggest table
        biggest = max(tables, key=lambda t: len(t) if t else 0)
        if len(biggest) > 5:  # real transaction table
            df = pd.DataFrame(biggest[1:], columns=biggest[0])
            return df, True
    
    text = page.extract_text()
    if text and ("balance" in text.lower() or "transaction" in text.lower() or "date" in text.lower()):
        return text, False
    return None, False

def process_page_with_ocr(page):
    pix = page.get_pixmap(matrix=fitz.Matrix(300/72))  # ~300 DPI
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    result = reader.readtext(img, detail=0, paragraph=True)
    return "\n".join(result)

def parse_text_to_transactions(text: str):
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    data = []
    date_pattern = re.compile(r'(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})')
    
    for line in lines:
        if date_pattern.search(line):
            parts = re.split(r'\s{2,}', line)
            parts = [p for p in parts if p]  # clean empty
            
            if len(parts) < 3:
                continue
                
            date = parts[0] if date_pattern.match(parts[0]) else None
            if not date:
                continue
                
            # Find all money values
            money_values = []
            for p in parts:
                cleaned = p.replace(',', '').replace(' ', '')
                if re.match(r'^[\-+]?[\d]+\.\d{2}$', cleaned):
                    money_values.append((p, float(cleaned)))
            
            if not money_values:
                continue
                
            # Standard pattern: last = balance, second last = amount (most common)
            amount_str = money_values[-2][0] if len(money_values) >= 2 else money_values[-1][0]
            
            # Build description from everything between date and amount
            desc = " ".join(parts[1:-1]) if len(money_values) >= 2 else " ".join(parts[1:])
            
            data.append([date, desc.strip(), amount_str.strip()])
    
    return pd.DataFrame(data, columns=["Date", "Description", "Amount"])

# Upload
uploaded_files = st.file_uploader("Upload bank statements (PDFs, scanned or digital)", accept_multiple_files=True, type=["pdf"])

if uploaded_files:
    all_dfs = []
    
    for file in uploaded_files:
        with st.spinner(f"Processing {file.name}..."):
            file_bytes = file.read()
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            file_dfs = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                result, is_table = process_page_with_plumber(pdfplumber.open(io.BytesIO(file_bytes)).pages[page_num])
                
                if is_table and isinstance(result, pd.DataFrame) and len(result) > 3:
                    df_page = result.copy()
                else:
                    # Fallback to OCR + text parsing
                    ocr_text = process_page_with_ocr(page)
                    df_page = parse_text_to_transactions(ocr_text)
                
                # Smart column handling
                cols = [c.lower() if isinstance(c, str) else "" for c in df_page.columns]
                date_col = next((i for i, c in enumerate(cols) if "date" in c or "trans" in c), 0)
                desc_col = next((i for i, c in enumerate(cols) if any(x in c for x in ["desc", "particular", "narr", "detail", "transact"]) , 1), None)
                
                if "debit" in cols and "credit" in cols:
                    debit_idx = cols.index("debit")
                    credit_idx = cols.index("credit")
                    df_page["Amount"] = df_page.iloc[:, credit_idx].fillna(0).astype(str).str.replace(',', '') .astype(float) - df_page.iloc[:, debit_idx].fillna(0).astype(str).str.replace(',', '').astype(float)
                    df_page["Amount"] = df_page["Amount"].apply(lambda x: f"{x:,.2f}" if abs(x) > 0 else "")
                    df_page["Description"] = df_page.iloc[:, desc_col] if desc_col is not None else ""
                elif "amount" in cols:
                    amount_idx = cols.index("amount")
                    df_page["Amount"] = df_page.iloc[:, amount_idx]
                    df_page["Description"] = df_page.iloc[:, desc_col] if desc_col is not None else df_page.iloc[:, 1]
                else:
                    df_page = parse_text_to_transactions(page.get_text() or process_page_with_ocr(page))
                
                df_page = df_page[["Date", "Description", "Amount"]].dropna(subset=["Amount"])
                df_page = df_page[df_page["Amount"] != "0.00"]
                if len(df_page) > 0:
                    file_dfs.append(df_page)
            
            if file_dfs:
                file_df = pd.concat(file_dfs, ignore_index=True)
                file_df.drop_duplicates(inplace=True)
                all_dfs.append(file_df)
    
    if all_dfs:
        final_df = pd.concat(all_dfs, ignore_index=True)
        final_df.sort_values("Date", inplace=True)
        final_df.reset_index(drop=True, inplace=True)
        
        st.success(f"✅ Extracted {len(final_df)} transactions!")
        st.dataframe(final_df, use_container_width=True)
        
        csv = final_df.to_csv(index=False).encode()
        st.download_button(
            "📄 Download Full CSV",
            data=csv,
            file_name="bank_statements_extracted.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.error("No transactions found — try another PDF or contact me for tweak")

st.markdown("---")
st.markdown("**Completely free · No API key ever · Runs EasyOCR locally on CPU · Works on scanned & digital PDFs**")
st.markdown("Share this link with your coworkers — unlimited use!")
