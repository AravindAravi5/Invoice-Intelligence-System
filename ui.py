"""
Streamlit Frontend for Invoice Intelligence System

This module provides a clean, professional user interface for the Invoice Intelligence System.
Users can upload invoice files (PDF/images) and view extracted information, clustering results,
and anomaly detection findings.

Features:
- File upload (PDF, PNG, JPG, JPEG, TIFF)
- Real-time processing with loading indicator
- Comprehensive results display
- Anomaly highlighting
- Error handling with user-friendly messages
"""

import streamlit as st
import requests
import os
from pathlib import Path
from datetime import datetime

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Invoice Intelligence System",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# STYLING
# ============================================================================

# Custom CSS for professional appearance and anomaly highlighting
st.markdown("""
    <style>
    /* Main container styling */
    .main {
        padding: 2rem;
    }
    
    /* Header styling */
    h1 {
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    h2 {
        color: #2c3e50;
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 0.5rem;
    }
    
    h3 {
        color: #34495e;
    }
    
    /* Anomaly highlight styling */
    .anomaly-box {
        background-color: #ffebee;
        border-left: 4px solid #d32f2f;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
    
    .anomaly-badge {
        color: white;
        background-color: #d32f2f;
        padding: 0.3rem 0.6rem;
        border-radius: 4px;
        font-weight: bold;
        font-size: 0.9rem;
    }
    
    .normal-badge {
        color: white;
        background-color: #388e3c;
        padding: 0.3rem 0.6rem;
        border-radius: 4px;
        font-weight: bold;
        font-size: 0.9rem;
    }
    
    /* Success message styling */
    .success-box {
        background-color: #e8f5e9;
        border-left: 4px solid #388e3c;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
    
    /* Field styling */
    .field-label {
        font-weight: bold;
        color: #2c3e50;
        margin-top: 0.5rem;
    }
    
    .field-value {
        color: #34495e;
        margin-left: 0.5rem;
        font-size: 1rem;
    }
    
    /* Results container */
    .results-container {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 8px;
        margin-top: 2rem;
    }
    
    /* Divider */
    hr {
        margin: 2rem 0;
        border: 0;
        border-top: 1px solid #e0e0e0;
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# CONFIGURATION
# ============================================================================

# API endpoint configuration - can be overridden with environment variable
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
PROCESS_ENDPOINT = f"{API_BASE_URL}/process-invoice"

# Supported file extensions
SUPPORTED_FORMATS = (".pdf", ".png", ".jpg", ".jpeg", ".tiff")

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

# Initialize session state variables to persist data across reruns
if "processed_result" not in st.session_state:
    st.session_state.processed_result = None

if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = None

if "is_loading" not in st.session_state:
    st.session_state.is_loading = False

# ============================================================================
# HEADER
# ============================================================================

st.markdown("# 📄 Invoice Intelligence System")
st.markdown("""
    **Powered by Advanced AI & Machine Learning**
    
    Upload your invoice document (PDF or Image) to extract information, identify clusters, 
    and detect anomalies automatically.
    """)

st.markdown("---")

# ============================================================================
# SIDEBAR - INFORMATION & SETTINGS
# ============================================================================

with st.sidebar:
    st.header("ℹ️ System Information")
    
    st.markdown("""
        ### About This Application
        
        This Invoice Intelligence System uses:
        - **OCR**: Text extraction from documents
        - **NER**: Named entity recognition for key fields
        - **Embeddings**: Vector representation of invoice data
        - **Clustering**: Grouping similar invoices
        - **Anomaly Detection**: Identifying suspicious invoices
        
        ### Supported File Formats
        - PDF Documents
        - PNG Images
        - JPG/JPEG Images
        - TIFF Images
        
        ### API Status
        """)
    
    # Check API health
    try:
        response = requests.get(f"{API_BASE_URL}/docs", timeout=2)
        if response.status_code == 200:
            st.success("✅ API Connected")
        else:
            st.warning("⚠️ API Status Unknown")
    except Exception as e:
        st.error(f"❌ API Unavailable\n\n{str(e)}")

# ============================================================================
# MAIN INTERFACE - UPLOAD SECTION
# ============================================================================

st.header("📤 Upload Invoice")

col1, col2 = st.columns([2, 1])

with col1:
    uploaded_file = st.file_uploader(
        "Choose an invoice file:",
        type=[fmt.strip(".") for fmt in SUPPORTED_FORMATS],
        help="Upload a PDF or image of your invoice document"
    )

with col2:
    st.markdown("")  # Spacing
    st.markdown("")  # Spacing
    process_button = st.button(
        "🚀 Process Invoice",
        use_container_width=True,
        type="primary",
        disabled=uploaded_file is None
    )

# ============================================================================
# PROCESSING LOGIC
# ============================================================================

if process_button and uploaded_file is not None:
    # Validate file extension
    file_extension = Path(uploaded_file.name).suffix.lower()
    
    if file_extension not in SUPPORTED_FORMATS:
        st.error(
            f"❌ **Unsupported File Format**\n\n"
            f"File '{uploaded_file.name}' has extension '{file_extension}'\n\n"
            f"Supported formats: {', '.join(SUPPORTED_FORMATS)}"
        )
    else:
        # Show loading spinner and processing message
        with st.spinner("⏳ Processing your invoice... This may take a moment."):
            try:
                # Prepare the file for upload
                files = {"file": (uploaded_file.name, uploaded_file.getbuffer(), uploaded_file.type)}
                
                # Send request to API
                response = requests.post(
                    PROCESS_ENDPOINT,
                    files=files,
                    timeout=60  # 60 second timeout for processing
                )
                
                # Handle API response
                if response.status_code == 200:
                    result = response.json()
                    
                    # Store result in session state
                    st.session_state.processed_result = result
                    st.session_state.uploaded_file_name = uploaded_file.name
                    
                    # Show success message
                    st.success(
                        f"✅ **Successfully processed:** {uploaded_file.name}\n\n"
                        f"Processed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                    )
                
                elif response.status_code == 400:
                    error_detail = response.json().get("detail", "Bad request")
                    st.error(f"❌ **Invalid Request**\n\n{error_detail}")
                
                elif response.status_code == 422:
                    error_detail = response.json().get("detail", "Processing error")
                    st.error(f"❌ **Processing Error**\n\n{error_detail}")
                
                else:
                    st.error(
                        f"❌ **API Error (Status {response.status_code})**\n\n"
                        f"{response.text}"
                    )
                    
            except requests.exceptions.Timeout:
                st.error(
                    "❌ **Request Timeout**\n\n"
                    "The processing took too long. Please try again or contact support."
                )
            except requests.exceptions.ConnectionError:
                st.error(
                    "❌ **Connection Error**\n\n"
                    "Cannot connect to the API server. Ensure the FastAPI backend is running "
                    "at " + API_BASE_URL
                )
            except Exception as e:
                st.error(f"❌ **Unexpected Error**\n\n{str(e)}")

# ============================================================================
# RESULTS SECTION
# ============================================================================

if st.session_state.processed_result is not None:
    result = st.session_state.processed_result
    
    st.markdown("---")
    st.header("📊 Processing Results")
    
    # Extract data from result
    ocr_data = result.get("pipeline_results", {}).get("ocr", {})
    nlp_data = result.get("pipeline_results", {}).get("nlp", {})
    ml_data = result.get("pipeline_results", {}).get("machine_learning", {})
    
    clustering_data = ml_data.get("clustering", {})
    anomaly_data = ml_data.get("anomaly_detection", {})
    
    structured_fields = nlp_data.get("structured_fields", {})
    is_anomaly = anomaly_data.get("is_anomaly", False)
    anomaly_score = anomaly_data.get("anomaly_score", 0.0)
    cluster_label = clustering_data.get("cluster_label", -1)
    
    # --- TAB 1: EXTRACTED FIELDS ---
    tab1, tab2, tab3 = st.tabs(["📝 Extracted Fields", "🔍 Raw Text", "⚙️ ML Analysis"])
    
    with tab1:
        st.subheader("Extracted Invoice Information")
        
        # Check if we have anomaly
        if is_anomaly:
            st.markdown(
                f"""
                <div class="anomaly-box">
                    <p style="margin: 0;">
                        <span class="anomaly-badge">⚠️ ANOMALY DETECTED</span>
                    </p>
                    <p style="margin-top: 0.5rem; margin-bottom: 0;">
                        This invoice has been flagged as suspicious. 
                        Anomaly Score: <strong>{anomaly_score:.4f}</strong>
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div class="success-box">
                    <p style="margin: 0;">
                        <span class="normal-badge">✓ NORMAL</span>
                    </p>
                    <p style="margin-top: 0.5rem; margin-bottom: 0;">
                        This invoice appears normal. 
                        Anomaly Score: <strong>{anomaly_score:.4f}</strong>
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        st.markdown("---")
        
        # Display extracted fields in organized layout
        if structured_fields:
            # Create columns for better layout
            col1, col2 = st.columns(2)
            
            # Define field grouping for better organization
            important_fields = ["vendor_name", "date", "total_amount", "invoice_number"]
            other_fields = {k: v for k, v in structured_fields.items() if k.lower() not in important_fields}
            
            # Important fields on the left
            with col1:
                for field in important_fields:
                    raw_val = structured_fields.get(field, "Not found")
                    # Handle both dict (from NER) and string formats
                    field_value = raw_val.get("value") if isinstance(raw_val, dict) else raw_val
                    
                    if field_value and str(field_value).strip().lower() != "not found":
                        st.markdown(f"<p class='field-label'>{field.upper()}</p>", unsafe_allow_html=True)
                        st.markdown(f"<p class='field-value'>{field_value}</p>", unsafe_allow_html=True)
            
            # Other fields on the right
            with col2:
                for field, raw_val in other_fields.items():
                    field_value = raw_val.get("value") if isinstance(raw_val, dict) else raw_val
                    
                    if field_value and str(field_value).strip().lower() != "not found":
                        st.markdown(f"<p class='field-label'>{field.upper()}</p>", unsafe_allow_html=True)
                        st.markdown(f"<p class='field-value'>{field_value}</p>", unsafe_allow_html=True)
        else:
            st.info("No structured fields were extracted from this document.")
    
    # --- TAB 2: RAW EXTRACTED TEXT ---
    with tab2:
        st.subheader("Raw OCR Text")
        
        text_snippet = ocr_data.get("extracted_text_snippet", "No text extracted")
        text_length = ocr_data.get("text_length", 0)
        
        st.info(f"📌 Total text length: **{text_length}** characters")
        st.markdown("---")
        
        # Display text in a code block for better readability
        st.text_area(
            "Extracted Text:",
            value=text_snippet,
            height=300,
            disabled=True,
            key="text_display"
        )
    
    # --- TAB 3: ML ANALYSIS ---
    with tab3:
        st.subheader("Machine Learning Analysis")
        
        # Clustering results
        st.markdown("#### 🎯 Clustering Results")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                "Cluster Label",
                f"{cluster_label}",
                help="Label assigned to this invoice by DBSCAN clustering"
            )
        
        with col2:
            cluster_desc = clustering_data.get("description", "Unknown")
            st.markdown(f"**Description:** {cluster_desc}")
        
        st.markdown("---")
        
        # Anomaly Detection results
        st.markdown("#### 🚨 Anomaly Detection Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Determine color based on anomaly status
            anomaly_status = "⚠️ ANOMALY" if is_anomaly else "✅ NORMAL"
            st.markdown(
                f"**Status:** <span class='{'anomaly-badge' if is_anomaly else 'normal-badge'}'>"
                f"{anomaly_status}</span>",
                unsafe_allow_html=True
            )
        
        with col2:
            st.metric(
                "Anomaly Score",
                f"{anomaly_score:.4f}",
                help="Score from Isolation Forest (higher = more anomalous)"
            )
        
        with col3:
            anomaly_desc = anomaly_data.get("description", "Unknown")
            st.markdown(f"**Status:** {anomaly_desc}")
        
        st.markdown("---")
        
        # Embedding information
        st.markdown("#### 🔢 Embedding Information")
        
        embedding_shape = ml_data.get("embedding_shape", [])
        if embedding_shape:
            st.info(
                f"Embedding shape: **{embedding_shape}**\n\n"
                f"The invoice data has been converted to a {embedding_shape[1]}-dimensional vector "
                f"for clustering and anomaly detection."
            )
        else:
            st.warning("Embedding information not available.")

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #888; font-size: 0.9rem;">
        <p>Invoice Intelligence System v1.0 | Powered by FastAPI & Streamlit</p>
        <p>© 2026 - Invoice & Financial Document Intelligence</p>
    </div>
    """, unsafe_allow_html=True)
