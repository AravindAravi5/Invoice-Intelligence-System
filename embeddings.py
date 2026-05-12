import json
import logging
import numpy as np
from typing import List, Union, Dict, Any
from sentence_transformers import SentenceTransformer

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class InvoiceEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the Embedding module.
        
        We use 'sentence-transformers' because they are specifically optimized to 
        produce semantically meaningful dense vectors that work exceptionally 
        well for downstream clustering tasks (like DBSCAN, HDBSCAN, or K-Means).
        
        Args:
            model_name (str): The name of the Hugging Face sentence-transformer model.
                              'all-MiniLM-L6-v2' is chosen by default for its excellent 
                              balance of high performance and low computational overhead.
        """
        logging.info(f"Loading embedding model: {model_name}...")
        try:
            self.model = SentenceTransformer(model_name)
            logging.info("Embedding model loaded successfully.")
        except Exception as e:
            logging.error(f"Failed to load the embedding model: {e}")
            raise

    def _prepare_text_from_json(self, invoice_data: Union[str, Dict[str, Any]]) -> str:
        """
        Converts structured JSON entity data into a coherent string representation.
        
        Why? Sentence transformers perform better on natural text representations rather 
        than raw JSON syntax (like brackets and quotes). We convert fields into sentences.
        
        Args:
            invoice_data: JSON string or dictionary of extracted entities.
            
        Returns:
            str: A formatted natural text string representing the invoice's core data.
        """
        # Parse JSON string if necessary
        if isinstance(invoice_data, str):
            try:
                data = json.loads(invoice_data)
            except json.JSONDecodeError:
                # If it's not valid JSON, treat it as raw text
                return invoice_data.strip()
        else:
            data = invoice_data
            
        # If it matches the nested format from our ner.py output, extract the 'entities'
        if "entities" in data:
            data = data["entities"]

        parts = []
        for key, value_dict in data.items():
            # Handle dictionary-style entity outputs containing 'value' and 'confidence'
            if isinstance(value_dict, dict) and 'value' in value_dict:
                val = value_dict['value']
            else:
                val = value_dict
                
            if val is not None:
                # Format into a readable sentence: "vendor_name" -> "Vendor Name: Acme Corp."
                clean_key = key.replace("_", " ").title()
                parts.append(f"{clean_key}: {val}.")
                
        return " ".join(parts)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Generate numerical vector embeddings for a batch of raw texts.
        
        Args:
            texts (List[str]): A list of string texts (e.g., raw OCR output).
            
        Returns:
            np.ndarray: A numerical matrix where each row is an embedding vector.
        """
        if not texts:
            raise ValueError("The input text list cannot be empty.")
            
        logging.info(f"Generating embeddings for {len(texts)} text(s)...")
        # encode() automatically handles batching and returns a numpy array
        embeddings = self.model.encode(texts, show_progress_bar=True)
        return embeddings

    def embed_json_batch(self, json_batch: List[Union[str, Dict[str, Any]]]) -> np.ndarray:
        """
        Generate embeddings for a batch of structured JSON invoices.
        This is ideal for clustering invoices based on their extracted entities.
        
        Args:
            json_batch (List[Union[str, Dict]]): A list of JSON strings or dicts.
            
        Returns:
            np.ndarray: A numerical matrix suitable for feeding directly into a clustering algorithm.
        """
        logging.info(f"Preparing {len(json_batch)} JSON records for embedding...")
        
        # 1. Convert all structured JSON records into semantic natural language strings
        prepared_texts = [self._prepare_text_from_json(record) for record in json_batch]
        
        # 2. Filter out any completely empty records to prevent errors
        valid_texts = [t for t in prepared_texts if t.strip()]
        
        if not valid_texts:
            raise ValueError("No valid text could be extracted from the provided JSON batch.")
            
        # 3. Generate and return the numerical embeddings
        return self.embed_texts(valid_texts)


# Example Usage for testing the module
if __name__ == "__main__":
    # Initialize the embedder (this will download the model on first run)
    print("Initializing Embedder...")
    embedder = InvoiceEmbedder()
    
    # --- Example 1: Embedding Raw OCR texts ---
    raw_texts = [
        "Acme Corp Invoice number 101 Total $500",
        "Globex Inc Invoice #552 Amount due 1200.00",
        "Acme Corporation INV-102 Total Amount: $550"
    ]
    
    print("\n--- Example 1: Embedding Raw Texts ---")
    raw_embeddings = embedder.embed_texts(raw_texts)
    # The shape will show: (Number of invoices, Dimensions of embedding [e.g., 384 for MiniLM])
    print(f"Generated embedding matrix shape: {raw_embeddings.shape}")
    
    # --- Example 2: Embedding Structured JSON Data (like from ner.py) ---
    json_invoices = [
        {
            "vendor_name": {"value": "Tech Solutions LLC", "confidence": 0.99},
            "total_amount": {"value": "$4,500.00", "confidence": 0.95}
        },
        {
            "vendor_name": {"value": "Office Supplies Co", "confidence": 0.88},
            "total_amount": {"value": "$125.50", "confidence": 0.91}
        }
    ]
    
    print("\n--- Example 2: Embedding Structured JSON Batch ---")
    json_embeddings = embedder.embed_json_batch(json_invoices)
    print(f"Generated embedding matrix shape: {json_embeddings.shape}")
