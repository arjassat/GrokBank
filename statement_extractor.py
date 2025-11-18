import streamlit as st
import pandas as pd
import tempfile
import io
import re
from tabula import read_pdf
import os
import requests
import json
import base64
from pdf2image import convert_from_bytes
from PyPDF2 import PdfReader 
import time
import random
import numpy as np

# --- 1. API Configuration and Schemas ---
# Note: This relies on a Streamlit Secret named 'GEMINI_API_KEY'
API_KEY = os.environ.get("GEMINI_API_KEY") 
API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent"

# JSON Schema for Transaction Extraction (CRITICAL for data quality)
TRANSACTION_SCHEMA = {
    "type": "ARRAY",
    "items": {
        "type": "OBJECT",
        "properties": {
            "Date": { "type": "STRING", "description": "The date of the transaction (YYYY-MM-DD)." },
            "Description": { "type": "STRING", "description": "The full transaction description." },
            "Amount": { "type": "NUMBER", "description": "The transaction amount. Debit/Withdrawals MUST be negative (-), Credits/Deposits MUST be positive (+)." }
        },
        "required": ["Date", "Description", "Amount"]
    }
}

# JSON Schema for Metadata Extraction (CRITICAL for automatic reconciliation)
METADATA_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "OpeningBalance": {"type": "NUMBER", "description": "The starting balance of the statement period."},
        "ClosingBalance": {"type": "NUMBER", "description": "The final closing balance of the statement period."},
        "StatementStartDate": {"type": "STRING", "description": "The start date of the statement period (YYYY-MM-DD)."},
        "StatementEndDate": {"type": "STRING", "description": "The end date of the statement period (YYYY-MM-DD)."}
    },
    "required": ["OpeningBalance", "ClosingBalance", "StatementStartDate", "StatementEndDate"]
}

# System Instruction for Transaction Extraction
TXN_SYSTEM_INSTRUCTION = """
You are a specialized financial transaction extractor for South African bank statements. Your task is to process raw bank statement text or images and strictly output a JSON array of transactions.

**CRITICAL INSTRUCTIONS:**
1. AGGRESSIVE EXTRACTION: Extract ALL lines that look like transactions. Every financial movement must be captured.
2. AMOUNT SIGN: Debits/Withdrawals must be negative. Credits/Deposits must be positive.
3. DATE IMPUTATION: If a transaction line does not contain an explicit date (e.g., in multi-line transactions), you MUST use the date from the immediately preceding transaction in the output JSON array.
4. FORMAT: Dates MUST be in YYYY-MM-DD format.

Strictly adhere to the provided JSON Schema for the output.
"""

# System Instruction for Metadata Extraction
META_SYSTEM_INSTRUCTION = """
You are a specialized financial header extractor. Your sole task is to find the opening balance, closing balance, statement start date, and statement end date from the provided document text or image.

**CRITICAL INSTRUCTIONS:**
1. Only extract the initial and final balances. Do not extract running balances.
2. Dates MUST be standardized to YYYY-MM-DD format.
3. Output MUST strictly adhere to the provided JSON Schema.
"""


# --- 2. Core LLM Utility Functions ---

def llm_api_call(payload, system_instruction, response_schema, is_image_data=False, page_num=None):
    """Handles API call with robust exponential backoff and jitter to prevent model overload."""
    full_api_url = f"{API_URL}?key={API_KEY}"
    max_retries = 5 # Increased retries for stability
    
    # Update system instruction in the payload
    payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}
    payload["generationConfig"] = {
        "responseMimeType": "application/json",
        "responseSchema": response_schema
    }
    
    for attempt in range(max_retries):
        try:
            headers = {'Content-Type': 'application/json'}
            response = requests.post(full_api_url, headers=headers, json=payload, timeout=90)
            response.raise_for_status() 

            result = response.json()
            
            if 'error' in result:
                raise Exception(f"API Error: {result['error']['message']}")
                
            json_string = result['candidates'][0]['content']['parts'][0]['text']
            return json.loads(json_string)

        except (requests.exceptions.RequestException, Exception) as e:
            error_label = "Network/API" if isinstance(e, requests.exceptions.RequestException) else "LLM"
            
            if attempt < max_retries - 1:
                # Exponential backoff (2^attempt) + Jitter (random delay)
                jitter = random.uniform(1.0, 4.0)
                wait_time = (2 ** attempt) + jitter
                
                # Only log/show wait time if it's an error that warrants retrying (503 or 429)
                if "429" in str(e) or "503" in str(e) or "API Error" in str(e):
                    if is_image_data:
                        st.warning(f"Page {page_num}: {error_label} error ({attempt+1}/{max_retries}). Retrying in {wait_time:.2f}s to prevent overload.")
                    else:
                        st.warning(f"Metadata: {error_label} error ({attempt+1}/{max_retries}). Retrying in {wait_time:.2f}s to prevent overload.")
                    
                    time.sleep(wait_time)
                    continue
                else:
                    # Non-retryable error (e.g., decoding failure)
                    st.error(f"Failed to process: {e}")
                    break
            else:
                st.error(f"Final attempt failed. Could not retrieve structured data due to repeated {error_label} errors.")
                raise Exception(f"Exceeded max retries for LLM call.")

    return None

def extract_metadata_with_llm(pdf_data, filename):
    """Extracts high-level balances and dates from the first page image/text."""
    st.info(f"Extracting crucial balances for '{filename}'...")
    
    # Strategy 1: Use raw text from PDF for fast extraction
    try:
        reader = PdfReader(io.BytesIO(pdf_data))
        raw_text = reader.pages[0].extract_text()
        
        payload = {
            "contents": [{"parts": [{"text": raw_text}]}],
        }
        
        metadata = llm_api_call(payload, META_SYSTEM_INSTRUCTION, METADATA_SCHEMA)
        if metadata:
            st.success("Successfully extracted statement metadata (Balances & Dates) via text.")
            return metadata
            
    except Exception as e:
        st.warning(f"Text-based metadata extraction failed: {e}. Falling back to image-based extraction.")
        
    # Strategy 2: Fallback to Image-based OCR on the first page
    try:
        images = convert_from_bytes(pdf_data, dpi=150)
        if not images: raise ValueError("Could not convert PDF to image.")

        img_byte_arr = io.BytesIO()
        images[0].save(img_byte_arr, format='JPEG')
        base64_image = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')

        payload = {
            "contents": [{
                "parts": [
                    {"inlineData": {"mimeType": "image/jpeg", "data": base64_image}},
                    {"text": "Extract the statement opening balance, closing balance, start date, and end date."}
                ]
            }],
        }
        
        metadata = llm_api_call(payload, META_SYSTEM_INSTRUCTION, METADATA_SCHEMA, is_image_data=True)
        if metadata:
            st.success("Successfully extracted statement metadata via image OCR.")
            return metadata
            
    except Exception as e:
        st.error(f"Failed to extract metadata via all methods: {e}")
        return None

def extract_transactions_with_llm(pdf_data, filename):
    """Converts each PDF page to image and uses LLM for transaction extraction."""
    st.info(f"Starting Multi-modal AI OCR Transaction Extraction for '{filename}'. This is the most robust step.")
    all_extracted_transactions = []
    
    try:
        images = convert_from_bytes(pdf_data, dpi=200) 
        total_pages = len(images)
        
        for page_num, image in enumerate(images):
            
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='JPEG')
            base64_image = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')

            user_query = "Extract all bank transactions from this page image into the required JSON format. Ensure all amounts are correctly signed (negative for debit, positive for credit)."
            
            payload = {
                "contents": [{
                    "parts": [
                        {"inlineData": {"mimeType": "image/jpeg", "data": base64_image}},
                        {"text": user_query}
                    ]
                }],
            }
            
            page_transactions = llm_api_call(payload, TXN_SYSTEM_INSTRUCTION, TRANSACTION_SCHEMA, is_image_data=True, page_num=page_num + 1)

            if page_transactions:
                all_extracted_transactions.extend(page_transactions)

            st.progress((page_num + 1) / total_pages, text=f"Processing page {page_num + 1} of {total_pages}...")

        return pd.DataFrame(all_extracted_transactions)

    except Exception as e:
        st.error(f"Critical error during transaction extraction (LLM method): {e}")
        return pd.DataFrame()


# --- 3. Main Processing Logic ---

def process_uploaded_files(uploaded_files):
    """Handles the main file processing loop (Metadata -> Transactions -> Reconciliation)."""
    
    final_transactions_df = pd.DataFrame(columns=['Date', 'Description', 'Amount'])
    all_metadata = []
    
    st.subheader("Extraction Progress")
            
    for uploaded_file in uploaded_files:
        st.write(f"--- Starting: **{uploaded_file.name}** ---")
        file_bytes = uploaded_file.getvalue()
        file_name = uploaded_file.name
        
        # Step 1: Automatic Metadata Extraction (Balances & Dates)
        metadata = extract_metadata_with_llm(file_bytes, file_name)
        if metadata:
            metadata['Filename'] = file_name
            all_metadata.append(metadata)
        else:
            st.warning(f"Skipping transaction extraction for '{file_name}' as critical balance metadata was not automatically found.")
            continue 

        # Step 2: AI OCR Transaction Extraction (Primary, Robust Method)
        df_llm = extract_transactions_with_llm(file_bytes, file_name)

        if df_llm.empty:
            st.warning("AI OCR extraction returned no data. Attempting Tabular Fallback...")
            # Step 2b: Tabular Fallback (for clean, native digital PDFs)
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(file_bytes)
                    temp_file_path = tmp.name
                    
                tabula_dfs = read_pdf(temp_file_path, pages='all', multiple_tables=True, stream=True, guess=True, silent=True)
                combined_tabula_df = pd.concat(tabula_dfs)
                
                # Crude column finding for fallback (less robust than AI)
                date_cols = [col for col in combined_tabula_df.columns if col and re.search(r'date|dat', col, re.IGNORECASE)]
                desc_cols = [col for col in combined_tabula_df.columns if col and re.search(r'description|narrative', col, re.IGNORECASE)]
                amt_cols = [col for col in combined_tabula_df.columns if col and re.search(r'amount|debit|credit|value', col, re.IGNORECASE)]

                if date_cols and desc_cols and amt_cols:
                    st.success(f"Successfully extracted tabular data via fallback for '{file_name}'.")
                    df = combined_tabula_df.copy()
                    
                    if len(amt_cols) >= 2: df['Amount'] = pd.to_numeric(df[amt_cols[1]].astype(str).str.replace(r'[^\d\.]', '', regex=True), errors='coerce') - pd.to_numeric(df[amt_cols[0]].astype(str).str.replace(r'[^\d\.]', '', regex=True), errors='coerce')
                    elif 'Amount' in df.columns or len(amt_cols) == 1: 
                        col = 'Amount' if 'Amount' in df.columns else amt_cols[0]
                        df['Amount'] = pd.to_numeric(df[col].astype(str).str.replace(r'[^\d\.\-]', '', regex=True), errors='coerce')
                    
                    df = df[[date_cols[0], desc_cols[0], 'Amount']].rename(columns={date_cols[0]: 'Date', desc_cols[0]: 'Description'})
                    df_llm = df[df['Amount'].abs() > 0].copy()
                else:
                    st.error(f"Tabular fallback failed for '{file_name}'. No transactions extracted.")
                
            except Exception as e:
                st.error(f"Tabular processing critical error for '{file_name}': {e}")
            finally:
                if temp_file_path and os.path.exists(temp_file_path): os.remove(temp_file_path)
        
        # Step 3: Final Data Cleanup and Merge
        if not df_llm.empty:
            df_llm['Filename'] = file_name
            df_llm['Amount'] = pd.to_numeric(df_llm['Amount'], errors='coerce')
            
            # Date Normalization and Imputation (Forward Fill)
            df_llm['Date'] = pd.to_datetime(df_llm['Date'], errors='coerce', dayfirst=True)
            df_llm['Date'] = df_llm['Date'].ffill() 
            df_llm['Date'] = df_llm['Date'].dt.strftime('%Y-%m-%d')
            
            final_transactions_df = pd.concat([final_transactions_df, df_llm.dropna(subset=['Amount'])], ignore_index=True)
            st.success(f"File '{file_name}' processed and added.")
        else:
            st.error(f"File '{file_name}' yielded no usable transactions.")


    st.markdown("---")
    st.header("Results Summary")
    st.balloons()
    
    # 4. Automatic Reconciliation Report Generation
    if all_metadata:
        report_data = []
        for meta in all_metadata:
            file_txns = final_transactions_df[final_transactions_df['Filename'] == meta['Filename']]
            total_change = file_txns['Amount'].sum()
            calculated_end_balance = meta['OpeningBalance'] + total_change
            
            diff = calculated_end_balance - meta['ClosingBalance']
            is_balanced = "✅ Balanced" if abs(diff) < 0.01 else "❌ Mismatch"
            
            report_data.append({
                "File": meta['Filename'],
                "Start Date": meta['StatementStartDate'],
                "End Date": meta['StatementEndDate'],
                "Opening Balance (Doc)": f"R{meta['OpeningBalance']:,.2f}",
                "Closing Balance (Doc)": f"R{meta['ClosingBalance']:,.2f}",
                "Total Transactions": f"R{total_change:,.2f}",
                "Calculated Closing Balance": f"R{calculated_end_balance:,.2f}",
                "Difference": f"R{diff:,.2f}",
                "Status": is_balanced
            })

        st.subheader("Automatic Reconciliation Report")
        st.dataframe(pd.DataFrame(report_data), hide_index=True, use_container_width=True)

    return final_transactions_df


# --- 4. Streamlit UI and Presentation ---

st.set_page_config(page_title="SA Bank Statement AI Extractor", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    /* Dark Theme inspired for financial data */
    .stApp {
        background-color: #1a1a2e; /* Dark Blue */
        color: #e6f7ff;
    }
    .main-header {
        font-size: 2.8em; 
        font-weight: 800; 
        color: #4CAF50; /* Green highlight */
        margin-bottom: 0.25em; 
        border-bottom: 3px solid #00BCD4; /* Cyan accent */
        padding-bottom: 10px;
    }
    .subheader {
        font-size: 1.3em; 
        color: #90caf9; /* Light blue */
        margin-bottom: 1.5em;
    }
    .stButton>button {
        background-color: #00BCD4; 
        color: white; 
        font-weight: 700; 
        border-radius: 0.75rem; 
        padding: 0.75rem 2rem; 
        box-shadow: 0 4px 10px rgba(0, 188, 212, 0.4); 
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #0097a7; 
        transform: translateY(-2px);
    }
    .stAlert {
        border-radius: 0.5rem;
    }
    /* Customize sidebar colors */
    [data-testid="stSidebar"] {
        background-color: #24243e; /* Slightly darker sidebar */
    }
    .sidebar-header {
        color: #4CAF50;
        font-weight: 600;
        font-size: 1.2em;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">🇿🇦 Bank Statement AI Extractor Pro</p>', unsafe_allow_html=True)
st.markdown('<p class="subheader">This tool uses multi-modal Gemini AI for robust OCR extraction, date imputation, and **automatic balance reconciliation**.</p>', unsafe_allow_html=True)

uploaded_files = st.file_uploader("Upload PDF Bank Statements (Digital or Scanned)", type=["pdf"], accept_multiple_files=True)

final_df = pd.DataFrame() # Initialize final_df

if uploaded_files:
    if st.button("🚀 Initiate Deep AI Extraction"):
        with st.spinner("Analyzing files, extracting balances, and processing transactions..."):
            final_df = process_uploaded_files(uploaded_files)

        if not final_df.empty:
            st.subheader("Extracted Transactions")
            st.dataframe(final_df[['Date', 'Description', 'Amount', 'Filename']], height=300, use_container_width=True)
            
            csv_output = final_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ Download All Transactions as CSV",
                data=csv_output,
                file_name="extracted_bank_statements_ai_reconciled.csv",
                mime="text/csv",
                key='download-csv-1'
            )
        else:
            st.error("No transactions could be extracted after all attempts. Please verify your files are clear and that your Gemini API Key is active.")

# --- Sidebar Status ---
st.sidebar.markdown('<p class="sidebar-header">🛠️ System Status & Logic</p>', unsafe_allow_html=True)
if API_KEY:
    st.sidebar.success("✅ **Gemini API Key Active!** AI Extraction fully enabled.")
else:
    st.sidebar.error("⚠️ **Gemini API Key Missing.** AI functions are disabled.")

st.sidebar.markdown("---")
st.sidebar.markdown("**Extraction Priority:**")
st.sidebar.markdown("1. **Automatic Metadata:** LLM extracts Opening/Closing Balances.")
st.sidebar.markdown("2. **AI-First Transactions:** LLM (OCR) performs page-by-page transaction extraction.")
st.sidebar.markdown("3. **Resilience:** Aggressive retries with **randomized backoff** prevent server overload.")
st.sidebar.markdown("4. **Reconciliation:** Checks if `Opening Balance + Transactions = Closing Balance`.")
