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

# --- API Configuration ---
# NOTE: This relies on a Streamlit Secret named 'GEMINI_API_KEY'
API_KEY = os.environ.get("GEMINI_API_KEY") 
API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent"
SYSTEM_INSTRUCTION_TEXT = """
You are a highly specialized financial transaction extractor. Your task is to process raw bank statement text or images and strictly output a JSON array of transactions.

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

def extract_data_from_pdf_image_with_llm_logic(pdf_data, filename):
    """
    Converts each PDF page to an image and sends it to the Gemini API 
    for OCR and structured extraction with robust error handling.
    """
    if not API_KEY:
        st.error("Gemini API Key is missing. Cannot perform image-based OCR extraction.")
        return pd.DataFrame()
        
    st.info(f"Processing '{filename}' page-by-page using **Gemini Multi-modal OCR Extraction**.")
    all_extracted_transactions = []
    
    try:
        images = convert_from_bytes(pdf_data, dpi=200) 
        total_pages = len(images)
        
        schema_definition = SYSTEM_INSTRUCTION_TEXT.split('Strictly adhere to this JSON Schema:')[1].strip()
        response_schema = json.loads(schema_definition)

        for page_num, image in enumerate(images):
            
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='JPEG')
            base64_image = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')

            user_query = "Extract all bank transactions from this page image into the required JSON format."
            
            payload = {
                "contents": [{
                    "parts": [
                        {
                            "inlineData": {
                                "mimeType": "image/jpeg",
                                "data": base64_image
                            }
                        },
                        {"text": user_query}
                    ]
                }],
                "systemInstruction": {"parts": [{"text": SYSTEM_INSTRUCTION_TEXT}]},
                "generationConfig": {
                    "responseMimeType": "application/json",
                    "responseSchema": response_schema
                }
            }
            
            # --- Make the API Call with Exponential Backoff and Error Handling ---
            max_retries = 3
            # Ensure the full URL is properly constructed
            full_api_url = f"{API_URL}?key={API_KEY}" 
            
            for attempt in range(max_retries):
                try:
                    headers = {'Content-Type': 'application/json'}
                    # Using the fully constructed URL
                    response = requests.post(full_api_url, headers=headers, json=payload, timeout=60)
                    response.raise_for_status() 

                    result = response.json()
                    
                    if 'error' in result:
                        st.error(f"API Error (Page {page_num + 1}, Attempt {attempt + 1}): {result['error']['message']}")
                        time.sleep(2 ** attempt) 
                        continue 

                    if 'candidates' not in result or not result.get('candidates'):
                        block_reason = result.get('promptFeedback', {}).get('blockReason', 'Content Blocked (Unknown Reason)')
                        st.warning(f"Content Blocked (Page {page_num + 1}): {block_reason}. Skipping page.")
                        break 

                    json_string = result['candidates'][0]['content']['parts'][0]['text']
                    
                    page_transactions = json.loads(json_string)
                    all_extracted_transactions.extend(page_transactions)
                    break 

                except requests.exceptions.RequestException as e:
                    # This is the section generating the "No connection adapters" error
                    st.error(f"API Request Error on page {page_num + 1} (Attempt {attempt + 1}): Network or HTTP failure: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)
                        continue
                    else:
                        break 
                except json.JSONDecodeError:
                    st.warning(f"Failed to decode JSON response on page {page_num + 1}. The model did not return valid JSON. Skipping page.")
                    break 
                except Exception as e:
                    st.error(f"Unexpected error processing page {page_num + 1}: {e}. Skipping page.")
                    break 
            
            st.progress((page_num + 1) / total_pages, text=f"Processing page {page_num + 1} of {total_pages}...")

        return pd.DataFrame(all_extracted_transactions)

    except Exception as e:
        st.error(f"Critical error during PDF-to-Image conversion: {e}. Check if 'packages.txt' is correctly installed with 'poppler-utils'.")
        return pd.DataFrame()


def process_uploaded_files(uploaded_files):
    """Handles the main file processing loop, trying table extraction first."""
    all_transactions_df = pd.DataFrame(columns=['Date', 'Description', 'Amount'])
    st.subheader("Extraction Progress")
            
    for uploaded_file in uploaded_files:
        st.write(f"--- Starting to process: **{uploaded_file.name}** ---")
        
        file_bytes = uploaded_file.getvalue()
        file_name = uploaded_file.name
        
        # 1. Try Tabular Extraction (Fastest for clean PDFs)
        temp_file_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file_bytes)
                temp_file_path = tmp.name
                
            tabula_dfs = read_pdf(temp_file_path, pages='all', multiple_tables=True, stream=True, guess=True, silent=True)
            combined_tabula_df = pd.concat(tabula_dfs)
            
            date_cols = [col for col in combined_tabula_df.columns if col and re.search(r'date|dat|txn|tranaction', col, re.IGNORECASE)]
            desc_cols = [col for col in combined_tabula_df.columns if col and re.search(r'description|narrative|details|particulars', col, re.IGNORECASE)]
            amt_cols = [col for col in combined_tabula_df.columns if col and re.search(r'amount|debit|credit|dr|cr|value', col, re.IGNORECASE)]

            if date_cols and desc_cols and amt_cols:
                st.success(f"Successfully extracted tabular data from '{file_name}'.")
                
                df = combined_tabula_df.copy()
                
                if len(amt_cols) >= 2: 
                    col1, col2 = amt_cols[0], amt_cols[1] 
                    df[col1] = pd.to_numeric(df[col1].astype(str).str.replace(r'[^\d\.]', '', regex=True), errors='coerce').fillna(0)
                    df[col2] = pd.to_numeric(df[col2].astype(str).str.replace(r'[^\d\.]', '', regex=True), errors='coerce').fillna(0)
                    df['Amount'] = df[col2] - df[col1] 
                elif 'Amount' in df.columns or len(amt_cols) == 1: 
                    col = 'Amount' if 'Amount' in df.columns else amt_cols[0]
                    df['Amount'] = pd.to_numeric(df[col].astype(str).str.replace(r'[^\d\.\-]', '', regex=True), errors='coerce')
                
                df = df[[date_cols[0], desc_cols[0], 'Amount']].rename(columns={date_cols[0]: 'Date', desc_cols[0]: 'Description'})
                df = df[df['Amount'].abs() > 0]
                all_transactions_df = pd.concat([all_transactions_df, df], ignore_index=True)
            else:
                raise ValueError("Inconsistent column structure found, falling back to AI OCR extraction.")
                
        except Exception as e:
            # 2. Fallback to Multi-modal AI OCR Extraction
            st.info(f"Tabular extraction failed for '{file_name}'. Initiating **AI OCR extraction** (page-by-page).")
            df_llm = extract_data_from_pdf_image_with_llm_logic(file_bytes, file_name)
            all_transactions_df = pd.concat([all_transactions_df, df_llm], ignore_index=True)
        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            
    # Final cleanup and normalization
    if not all_transactions_df.empty:
        all_transactions_df = all_transactions_df.drop_duplicates().reset_index(drop=True)
        all_transactions_df['Amount'] = pd.to_numeric(all_transactions_df['Amount'].astype(str).str.replace(r'[^\d\.\-]', '', regex=True), errors='coerce')
        all_transactions_df = all_transactions_df.dropna(subset=['Amount'])
        
        try:
            all_transactions_df['Date'] = pd.to_datetime(all_transactions_df['Date'], errors='coerce')
            all_transactions_df = all_transactions_df.sort_values(by='Date').reset_index(drop=True)
            all_transactions_df['Date'] = all_transactions_df['Date'].dt.strftime('%Y-%m-%d')
        except:
            st.warning("Could not reliably convert 'Date' column to a standard format for sorting. Dates are raw text.")

        st.balloons()
        st.success("✅ **Extraction Complete!** Download your CSV below.")

    return all_transactions_df

# --- Streamlit UI ---

st.set_page_config(page_title="Free Bank Statement AI Extractor", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .stApp {background-color: #f7f9fc;}
    .main-header {font-size: 2.5em; font-weight: 700; color: #1e3a8a; margin-bottom: 0.25em; border-bottom: 3px solid #3b82f6; padding-bottom: 10px;}
    .subheader {font-size: 1.25em; color: #4b5563; margin-bottom: 1.5em;}
    .stButton>button {background-color: #3b82f6; color: white; font-weight: 600; border-radius: 0.5rem; padding: 0.75rem 1.5rem; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -2px rgba(0, 0, 0, 0.1); transition: background-color 0.3s;}
    .stButton>button:hover {background-color: #2563eb;}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">📄 Free Bank Statement AI Extractor</p>', unsafe_allow_html=True)
st.markdown('<p class="subheader">Upload your PDF or scanned bank statements. Uses **Tabular** for clean PDFs and **Gemini AI OCR** for scanned documents.</p>', unsafe_allow_html=True)

uploaded_files = st.file_uploader("Upload Bank Statements (PDF/Scanned Images)", type=["pdf"], accept_multiple_files=True)

# Add a check for the API key being available in the environment 
if API_KEY:
    api_key_status = "✅ **Gemini API Key Found!** Multi-modal OCR extraction is enabled."
else:
    api_key_status = "⚠️ **Gemini API Key Missing.** Multi-modal OCR extraction (required for scanned files) is disabled."


if uploaded_files:
    if st.button("🚀 Start AI Extraction"):
        with st.spinner("Analyzing files and extracting transactions..."):
            final_df = process_uploaded_files(uploaded_files)
        
        if not final_df.empty:
            st.subheader("Final Extracted Transactions")
            st.dataframe(final_df, height=300, use_container_width=True)
            
            csv_output = final_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ Download Transactions as CSV",
                data=csv_output,
                file_name="extracted_bank_statements.csv",
                mime="text/csv",
                key='download-csv-1'
            )
        else:
            st.error("No transactions could be extracted. Please ensure your files are clear, text-readable PDFs or high-quality scanned copies, and verify your API Key is valid.")

st.sidebar.header("API Key Status")
st.sidebar.markdown(api_key_status)
if not API_KEY:
    st.sidebar.caption("Please add your key to Streamlit secrets as `GEMINI_API_KEY`.")
st.sidebar.header("Extraction Logic")
st.sidebar.markdown("1. Tries **Tabular extraction** (fastest for digital PDFs).")
st.sidebar.markdown("2. If that fails, it uses **Gemini AI OCR** (for scanned images/difficult layouts).")
