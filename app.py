import os
import json
import uuid
import logging
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

# Import our custom AI pipeline modules
from ocr import OCRExtractor
from ner import InvoiceNER
from embeddings import InvoiceEmbedder
from clustering import InvoiceClusterer
from anomaly import InvoiceAnomalyDetector

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Get system dependency paths from environment variables (optional)
TESSERACT_PATH = os.getenv('TESSERACT_PATH', None)
POPPLER_PATH = os.getenv('POPPLER_PATH', None)

if TESSERACT_PATH:
    logging.info(f"Using custom Tesseract path: {TESSERACT_PATH}")
if POPPLER_PATH:
    logging.info(f"Using custom Poppler path: {POPPLER_PATH}")

# Initialize FastAPI app
app = FastAPI(
    title="Invoice Intelligence API",
    description="AI-powered backend for processing, analyzing, and clustering financial documents.",
    version="1.0.0"
)

# Initialize ML models globally. 
# Loading them on startup prevents the API from crashing during a request and speeds up response times.
logging.info("Initializing ML Models (This may take a moment)...")
try:
    ocr_extractor = OCRExtractor(tesseract_cmd=TESSERACT_PATH, poppler_path=POPPLER_PATH)
    ner_extractor = InvoiceNER()
    embedder = InvoiceEmbedder()
    
    # Note: For a true production system, clustering and anomaly detection are usually fitted on 
    # historical database data, and new invoices are compared against that fitted model. 
    # For this endpoint, we initialize them here to demonstrate the full data flow.
    clusterer = InvoiceClusterer(eps=0.1, min_samples=3)
    anomaly_detector = InvoiceAnomalyDetector(contamination=0.1)
    
    logging.info("All ML models loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load ML models on startup: {e}")
    # We log the error but don't strictly raise, allowing the server to boot for debugging, 
    # though endpoints will fail if models are missing.

# Ensure a temporary directory exists for uploads
UPLOAD_DIR = "temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/process-invoice")
async def process_invoice(file: UploadFile = File(...)):
    """
    Main endpoint to process an uploaded invoice through the full AI pipeline.
    
    Data Flow:
    Upload -> OCR (Text) -> NER (JSON) -> Embedder (Vector) -> DBSCAN (Cluster) -> Isolation Forest (Anomaly)
    """
    # 1. Validate File format
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.pdf', '.tiff')):
        raise HTTPException(status_code=400, detail="Unsupported file format. Use PDF or Images.")
        
    # Generate a unique filename and save locally for processing
    file_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}_{file.filename}")
    
    try:
        # Save the uploaded file to disk
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())
            
        logging.info(f"Processing started for file: {file.filename}")

        # --- STEP 1: OCR Text Extraction ---
        logging.info("Step 1: Running OCR...")
        try:
            raw_text = ocr_extractor.extract_text(file_path)
            if not raw_text.strip():
                raise ValueError("OCR returned empty text.")
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"OCR Processing failed: {str(e)}")

        # --- STEP 2: NLP/NER Entity Extraction ---
        # The NER module asks QA questions to extract vendor, date, amount, etc.
        logging.info("Step 2: Running NER...")
        try:
            ner_json_str = ner_extractor.extract_entities(raw_text)
            ner_data = json.loads(ner_json_str)
            
            if ner_data.get("status") == "error":
                raise ValueError(ner_data.get("message", "Unknown NER error"))
                
            extracted_entities = ner_data.get("entities", {})
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"NER Processing failed: {str(e)}")

        # --- STEP 3: Generate Embeddings ---
        # Converts the structured JSON into a dense numerical vector (e.g., 1x384 array)
        logging.info("Step 3: Generating Embeddings...")
        try:
            # We pass it as a batch of 1
            embeddings = embedder.embed_json_batch([extracted_entities])
            embedding_shape = list(embeddings.shape)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Embedding generation failed: {str(e)}")

        # --- STEP 4: Clustering ---
        # Assign this invoice to a cluster.
        # Note: Since we are only passing 1 embedding, DBSCAN will likely return -1 (Noise)
        # because the default min_samples is 3. In production, this embedding would be appended 
        # to historical data before clustering.
        logging.info("Step 4: Running Clustering...")
        try:
            cluster_labels = clusterer.fit_predict(embeddings)
            cluster_label = int(cluster_labels[0])
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Clustering failed: {str(e)}")

        # --- STEP 5: Anomaly Detection ---
        # Check if this invoice is statistically anomalous compared to the model.
        logging.info("Step 5: Running Anomaly Detection...")
        try:
            anomaly_results = anomaly_detector.fit_predict(embeddings, is_1d_amounts=False)
            anomaly_label = int(anomaly_results["labels"][0])
            anomaly_score = float(anomaly_results["scores"][0])
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Anomaly detection failed: {str(e)}")

        # --- STEP 6: Construct Final JSON Response ---
        logging.info("Pipeline completed successfully.")
        
        response_payload = {
            "status": "success",
            "filename": file.filename,
            "pipeline_results": {
                "ocr": {
                    # Return a snippet of the text to keep the JSON clean, plus the full length
                    "extracted_text_snippet": raw_text[:300] + "..." if len(raw_text) > 300 else raw_text,
                    "text_length": len(raw_text)
                },
                "nlp": {
                    "structured_fields": extracted_entities
                },
                "machine_learning": {
                    "embedding_shape": embedding_shape,
                    "clustering": {
                        "cluster_label": cluster_label,
                        "description": "Noise / Unique Invoice" if cluster_label == -1 else f"Grouped in Cluster {cluster_label}"
                    },
                    "anomaly_detection": {
                        "is_anomaly": bool(anomaly_label == -1),
                        "anomaly_score": round(anomaly_score, 4),
                        "description": "Highly Suspicious" if anomaly_label == -1 else "Normal"
                    }
                }
            }
        }
        
        return JSONResponse(content=response_payload)
        
    finally:
        # Cleanup: Always remove the temporary uploaded file to prevent disk bloat
        if os.path.exists(file_path):
            os.remove(file_path)

if __name__ == "__main__":
    import uvicorn
    # Run the server locally on port 8000
    print("Starting FastAPI server on http://localhost:8000 ...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
