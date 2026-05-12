import json
import logging
from typing import Dict, Any
from transformers import pipeline

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class InvoiceNER:
    def __init__(self, model_name: str = "deepset/roberta-base-squad2"):
        """
        Initialize the NLP Entity Extraction module using AutoModel/Tokenizer.
        """
        logging.info(f"Loading transformer model: {model_name}...")
        try:
            from transformers import AutoTokenizer, AutoModelForQuestionAnswering
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
            logging.info("Model and Tokenizer loaded successfully.")
        except Exception as e:
            logging.error(f"Failed to load the model: {e}")
            raise

    def extract_entities(self, text: str, confidence_threshold: float = 0.1) -> str:
        """
        Extract key invoice fields from the given text using the transformer model.
        
        Args:
            text (str): The raw text extracted from the invoice (e.g., via OCR).
            confidence_threshold (float): Minimum confidence score to accept an extraction.
            
        Returns:
            str: A JSON formatted string containing the structured entities and their scores.
        """
        # Handle empty text cases gracefully
        if not text or not text.strip():
            logging.warning("Empty text provided for extraction.")
            return json.dumps({"status": "error", "message": "No text provided"}, indent=4)

        # Define the fields we want to extract and the questions to ask the model
        queries = {
            "vendor_name": "What is the name of the vendor or company?",
            "invoice_number": "What is the invoice number?",
            "date": "What is the invoice date?",
            "total_amount": "What is the total amount due or paid?"
        }

        extracted_data = {}

        logging.info("Starting entity extraction...")
        
        for field, question in queries.items():
            try:
                # Manual inference for Question Answering
                import torch
                inputs = self.tokenizer(question, text, return_tensors="pt", max_length=512, truncation=True)
                with torch.no_grad():
                    outputs = self.model(**inputs)

                answer_start = torch.argmax(outputs.start_logits)
                answer_end = torch.argmax(outputs.end_logits) + 1

                # Calculate confidence score
                start_prob = torch.max(torch.softmax(outputs.start_logits, dim=-1))
                end_prob = torch.max(torch.softmax(outputs.end_logits, dim=-1))
                score = ((start_prob + end_prob) / 2.0).item()

                answer = self.tokenizer.decode(inputs.input_ids[0][answer_start:answer_end], skip_special_tokens=True).strip()
                
                # Check if the model's confidence meets our threshold
                if score >= confidence_threshold and answer:
                    extracted_data[field] = {
                        "value": answer,
                        "confidence": round(score, 4)
                    }
                else:
                    # Handle low confidence or missing entities
                    extracted_data[field] = {
                        "value": None,
                        "confidence": round(score, 4),
                        "note": "Confidence too low or entity not found"
                    }
                    
            except Exception as e:
                # Catch any unexpected errors for a specific field so the rest can still process
                logging.error(f"Error extracting field '{field}': {e}")
                extracted_data[field] = {
                    "value": None,
                    "confidence": 0.0,
                    "error": str(e)
                }

        # Structure the final output nicely
        final_output = {
            "status": "success",
            "entities": extracted_data
        }

        # Return as a cleanly formatted JSON string
        return json.dumps(final_output, indent=4)


# Example Usage for testing the module
if __name__ == "__main__":
    # Sample OCR text from a hypothetical invoice
    sample_ocr_text = """
    Acme Corp
    123 Business Rd, Metropolis
    
    INVOICE
    Invoice Number: INV-98765
    Date: October 24, 2023
    
    Bill To:
    John Doe
    
    Description                 Amount
    ----------------------------------
    Consulting Services         $1,500.00
    Software License            $500.00
    ----------------------------------
    Total Amount Due:           $2,000.00
    """
    
    print("Initializing NER Extractor (this may download the model on first run)...")
    ner_extractor = InvoiceNER()
    
    print("\n--- Extracting Entities from Sample Text ---\n")
    json_result = ner_extractor.extract_entities(sample_ocr_text)
    
    print(json_result)
