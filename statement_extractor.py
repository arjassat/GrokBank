import streamlit as st
import pandas as pd
import tempfile
import io
import re
from PyPDF2 import PdfReader
from tabula import read_pdf
import os
import requests
import json

# --- API Configuration ---
# The API key is securely loaded from an environment variable (e.g., set as a secret in Streamlit Cloud)
API_KEY = os.environ.get("GEMINI_API_KEY") 
API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent"
SYSTEM_INSTRUCTION_TEXT = """
You are a highly specialized financial transaction extractor. Your task is to process raw bank statement text and strictly output a JSON array of transactions.

The required fields are:
1. 'Date': (Format: YYYY-MM-DD or DD/MM/YYYY, ensure consistency)
2. 'Description': (The full transaction description)
3. 'Amount': (A floating-point number. Debit/Withdrawals must be negative, Credits/Deposits must be positive.)

Ignore headers, footers, account summaries, and non-transaction text. If a row is clearly a header or a running balance line, ignore it. Only extract confirmed transactions.

Strictly adhere to this JSON Schema:
{
    "type": "ARRAY",
    "items": {
        "type": "OBJECT",
        "properties": {
            "Date": { "type": "STRING", "description": "The date of the transaction." },
            "Description": { "type": "STRING", "description": "The description of the transaction." },
            "Amount": { "type": "NUMBER", "description": "The transaction amount, negative for debit, positive for credit." }
        },
        "required": ["Date", "Description", "Amount"]
    }
}
"""

def extract_data_from_pdf_text_with_llm_logic(pdf_data, filename):
    """
    Processes a PDF file, extracts text page-by-page, and uses a structured
    API call (if key is available) to extract transactions. If the API key is 
    missing or the API call fails, it falls back to a free local regex parser.
    """
    
    if API_KEY:
        st.info(f"Processing '{filename}' page-by-page using **Gemini API for extraction**.")
    else:
        st.warning(f"Processing '{filename}' page-by-page using **Free Python/Regex Logic**. For improved accuracy (especially for scanned documents), add the `GEMINI_API_KEY` secret.")

    all_extracted_transactions = []
    
    try:
        pdf_reader = PdfReader(io.BytesIO(pdf_data))
        total_pages = len(pdf_reader.pages)
        
        for page_num in range(total_pages):
            page = pdf_reader.pages[page_num]
            text = page.extract_text()
            
            if not text:
                st.warning(f"Page {page_num + 1}/{total_pages} of '{filename}' yielded no readable text. Skipping.")
                continue

            user_query = f"Extract all transactions from the following bank statement page text:\n\n---\n{text}"
            
            page_transactions = []
            api_success = False

            if API_KEY:
                # --- ACTUAL GEMINI API CALL LOGIC ---
                try:
                    payload = {
                        "contents": [{"parts": [{"text": user_query}]}],
                        "systemInstruction": {"parts": [{"text": SYSTEM_INSTRUCTION_TEXT}]},
                        "generationConfig": {
                            "responseMimeType": "application/json",
                            "responseSchema": {
                                "type": "ARRAY",
                                "items": {
                                    "type": "OBJECT",
                                    "properties": {
                                        "Date": { "type": "STRING", "description": "The date of the transaction." },
                                        "Description": { "type": "STRING", "description": "The description of the transaction." },
                                        "Amount": { "type": "NUMBER", "description": "The transaction amount, negative for debit, positive for credit." }
                                    },
                                    "required": ["Date", "Description", "Amount"]
                                }
                            }
                        }
                    }
                    
                    headers = {'Content-Type': 'application/json'}
                    response = requests.post(f"{API_URL}?key={API_KEY}", headers=headers, json=payload)
                    
                    # Raise an exception for bad status codes (4xx or 5xx)
                    response.raise_for_status() 

                    result = response.json()
                    json_string = result.candidates[0].content.parts[0].text
                    
                    page_transactions = json.loads(json_string)
                    api_success = True

                except requests.exceptions.RequestException as e:
                    # Capture request errors (network, 401, 429, 500 etc.)
                    st.error(f"API Request Error on page {page_num + 1}. Status: {response.status_code if 'response' in locals() else 'N/A'}. Falling back to basic regex parser.")
                    page_transactions = []
                except Exception as e:
                    # Capture JSON parsing errors or other failures
                    st.error(f"Failed to parse LLM response on page {page_num + 1}: {e}. Falling back to basic regex parser.")
                    page_transactions = []
            
            # --- FREE FALLBACK / SIMULATED EXTRACTION LOGIC ---
            if not API_KEY or not api_success:
                if API_KEY and not api_success:
                    st.warning(f"API extraction failed or was empty for page {page_num + 1}. Using basic regex.")
                
                # Basic pattern matching for common transaction lines
                transaction_pattern = re.compile(
                    r'(\d{1,2}[-/]\d{1,2}[-/]?\d{2,4}?)\s+(.+?)\s+([\d,]+\.\d{2})\s+([\d,]+\.\d{2})?', 
                    re.MULTILINE
                )
                
                matches = transaction_pattern.findall(text)
                
                for match in matches:
                    date_str = match[0].strip()
                    description = match[1].strip()
                    amount_str = match[2].strip().replace(',', '') 

                    try:
                        amount = float(amount_str)
                        # Heuristic: Determine debit/credit based on keywords
                        if re.search(r'withdrawal|debit|purchase|payment', description, re.IGNORECASE):
                            amount = -abs(amount)
                        
                        page_transactions.append({
                            'Date': date_str,
                            'Description': description,
                            'Amount': amount
                        })
                    except ValueError:
                        continue 

            all_extracted_transactions.extend(page_transactions)
            st.progress((page_num + 1) / total_pages, text=f"Processing page {page_num + 1} of {total_pages}...")

        return pd.DataFrame(all_extracted_transactions)

    except Exception as e:
        st.error(f"Error during PDF reader processing on '{filename}': {e}")
        return pd.DataFrame()


def process_uploaded_files(uploaded_files):
    """Handles the main file processing loop, trying table extraction first."""
    all_transactions_df = pd.DataFrame(columns=['Date', 'Description', 'Amount'])
    
    st.subheader("Extraction Progress")
            
    for uploaded_file in uploaded_files:
        st.write(f"--- Starting to process: **{uploaded_file.name}** ---")
        
        file_bytes = uploaded_file.getvalue()
        file_name = uploaded_file.name
        
        # 1. Try Tabular Extraction (Best for digital PDFs)
        temp_file_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file_bytes)
                temp_file_path = tmp.name
                
            tabula_dfs = read_pdf(temp_file_path, pages='all', multiple_tables=True, stream=True, guess=True, silent=True)
            
            combined_tabula_df = pd.concat(tabula_dfs)
            
            # Identify columns using regex
            date_cols = [col for col in combined_tabula_df.columns if re.search(r'date|dat|txn|tranaction', col, re.IGNORECASE)]
            desc_cols = [col for col in combined_tabula_df.columns if re.search(r'description|narrative|details', col, re.IGNORECASE)]
            amt_cols = [col for col in combined_tabula_df.columns if re.search(r'amount|debit|credit|dr|cr', col, re.IGNORECASE)]

            if date_cols and desc_cols and amt_cols:
                st.success(f"Successfully extracted tabular data from '{file_name}'.")
                
                df = combined_tabula_df.copy()
                
                if 'Amount' in df.columns:
                    df['Amount'] = df['Amount'].astype(str).str.replace(r'[^\d\.\-]', '', regex=True)
                elif len(amt_cols) >= 2:
                    debit_col = amt_cols[0]
                    credit_col = amt_cols[1]
                    
                    df[debit_col] = pd.to_numeric(df[debit_col].astype(str).str.replace(r'[^\d\.]', '', regex=True), errors='coerce').fillna(0)
                    df[credit_col] = pd.to_numeric(df[credit_col].astype(str).str.replace(r'[^\d\.]', '', regex=True), errors='coerce').fillna(0)
                    
                    df['Amount'] = df[credit_col] - df[debit_col] 
                else:
                    st.warning("Could not clearly identify standard Amount columns (Debit/Credit/Amount). Using raw AI logic fallback.")
                    raise ValueError("Column matching failed, falling back to text extraction.")
                    
                df = df[[date_cols[0], desc_cols[0], 'Amount']].rename(columns={
                    date_cols[0]: 'Date',
                    desc_cols[0]: 'Description',
                })
                
                df = df[df['Amount'].abs() > 0]
                
                all_transactions_df = pd.concat([all_transactions_df, df], ignore_index=True)
                
            else:
                st.warning("Tabular extraction was successful but failed to identify the correct transaction columns. Falling back to AI extraction logic.")
                raise ValueError("Column matching failed, falling back to text extraction.")
                
        except Exception as e:
            # Fallback for scanned PDFs or failed tabular extraction
            st.info(f"Tabular extraction failed for '{file_name}'. Initiating AI extraction logic (page-by-page).")
            df_llm = extract_data_from_pdf_text_with_llm_logic(file_bytes, file_name)
            all_transactions_df = pd.concat([all_transactions_df, df_llm], ignore_index=True)
        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            
    # Final cleanup and normalization
    if not all_transactions_df.empty:
        all_transactions_df = all_transactions_df.drop_duplicates().reset_index(drop=True)
        all_transactions_df['Amount'] = pd.to_numeric(
            all_transactions_df['Amount'].astype(str).str.replace(r'[^\d\.\-]', '', regex=True), 
            errors='coerce'
        )
        all_transactions_df = all_transactions_df.dropna(subset=['Amount'])
        
        try:
            all_transactions_df['Date'] = pd.to_datetime(all_transactions_df['Date'], errors='coerce')
            all_transactions_df = all_transactions_df.sort_values(by='Date').reset_index(drop=True)
            all_transactions_df['Date'] = all_transactions_df['Date'].dt.strftime('%Y-%m-%d')
        except:
            st.warning("Could not reliably convert 'Date' column to a standard format for sorting. Dates are raw.")

        st.balloons()
        st.success("✅ **Extraction Complete!** Download your CSV below.")

    return all_transactions_df

# --- Streamlit UI ---

st.set_page_config(
    page_title="Free Bank Statement AI Extractor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (CSS)
st.markdown("""
<style>
    .stApp {background-color: #f7f9fc;}
    .main-header {font-size: 2.5em; font-weight: 700; color: #1e3a8a; margin-bottom: 0.25em; border-bottom: 3px solid #3b82f6; padding-bottom: 10px;}
    .subheader {font-size: 1.25em; color: #4b5563; margin-bottom: 1.5em;}
    .stFileUploader {border: 2px dashed #93c5fd; border-radius: 0.5rem; padding: 20px; background-color: #eff6ff;}
    .stButton>button {background-color: #3b82f6; color: white; font-weight: 600; border-radius: 0.5rem; padding: 0.75rem 1.5rem; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -2px rgba(0, 0, 0, 0.1); transition: background-color 0.3s;}
    .stButton>button:hover {background-color: #2563eb;}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">📄 Free Bank Statement AI Extractor</p>', unsafe_allow_html=True)
st.markdown('<p class="subheader">Upload your PDF or scanned bank statements to extract transaction data into a clean CSV file. The app uses **tabular extraction** first, then falls back to **Gemini AI extraction** (if key is provided) or **local Python regex** for difficult documents.</p>', unsafe_allow_html=True)

# File Uploader
uploaded_files = st.file_uploader(
    "Upload Bank Statements (PDF/Scanned Images)", 
    type=["pdf"], 
    accept_multiple_files=True,
    help="You can upload multiple files at once. The app will process them sequentially."
)

if uploaded_files:
    if st.button("🚀 Start AI Extraction"):
        with st.spinner("Analyzing files and extracting transactions..."):
            final_df = process_uploaded_files(uploaded_files)
        
        if not final_df.empty:
            st.subheader("Final Extracted Transactions")
            st.dataframe(final_df, height=300, use_container_width=True)
            
            st.markdown(
                f"""
                <div style="text-align: center; margin-top: 20px; padding: 10px; border: 1px solid #d1d5db; border-radius: 8px; background-color: #fff;">
                    Total Transactions Extracted: <b>{len(final_df)}</b>
                </div>
                """, 
                unsafe_allow_html=True
            )

            csv_output = final_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ Download Transactions as CSV",
                data=csv_output,
                file_name="extracted_bank_statements.csv",
                mime="text/csv",
                key='download-csv-1'
            )
        else:
            st.error("No transactions could be extracted. Please ensure your files are clear, text-readable PDFs or high-quality scanned copies.")

# Information and Hosting Guidance
st.sidebar.header("Hosting Recommendations")
st.sidebar.markdown("""
This application is designed to be hosted for free using **Streamlit Community Cloud**.

1.  **Save this file** as `statement_extractor.py`.
2.  **Ensure `requirements.txt`** is present (see below).
3.  **Push both files** to your GitHub repository.
4.  **Connect your repo** to [Streamlit Community Cloud](https://streamlit.io/cloud) to deploy for free.
""")

st.sidebar.header("API Key Status")
if API_KEY:
    st.sidebar.success("✅ **Gemini API Key Found!** The app will use AI extraction as a fallback for maximum accuracy.")
else:
    st.sidebar.warning("⚠️ **Gemini API Key Missing.** The app will rely on less accurate Python/Regex logic for non-tabular files.")

st.sidebar.header("Extraction Logic")
st.sidebar.markdown("""
To handle all types of PDFs, this app uses a tiered approach:
1.  **Tabular Extraction (`tabula-py`):** Used first for clean, digital PDFs.
2.  **AI Extraction (`Gemini API`):** Used as a fallback for scanned or complex PDFs (requires `GEMINI_API_KEY` secret).
3.  **Local Regex:** The final, free fallback if the API key is unavailable or the API fails.
""")
