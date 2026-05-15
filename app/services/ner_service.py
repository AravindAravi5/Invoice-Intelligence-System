"""
NER Service — Named Entity Recognition for Invoice Fields
===========================================================
Uses a pre-trained Question-Answering transformer model (RoBERTa-SQuAD2)
to extract structured fields from raw invoice text.

Approach:
    For each target field (vendor, date, amount, invoice number), we pose a
    natural language question to the QA model. The model then identifies the
    answer span within the OCR text.

Why QA-based NER?
    Traditional NER models are trained on generic entities (PERSON, ORG, DATE).
    Invoice-specific fields like "invoice number" or "total amount" are better
    captured by asking targeted questions against the document context.
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, Optional

import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

from app.exceptions import NERExtractionError

logger = logging.getLogger(__name__)


# ── Data Structures ──────────────────────────────────────────────────────

@dataclass
class ExtractedField:
    """A single extracted invoice field with its confidence score."""
    value: Optional[str]
    confidence: float
    note: Optional[str] = None


@dataclass
class NERResult:
    """Complete NER extraction result for one invoice."""
    vendor_name: ExtractedField = field(default_factory=lambda: ExtractedField(None, 0.0))
    invoice_number: ExtractedField = field(default_factory=lambda: ExtractedField(None, 0.0))
    date: ExtractedField = field(default_factory=lambda: ExtractedField(None, 0.0))
    total_amount: ExtractedField = field(default_factory=lambda: ExtractedField(None, 0.0))

    def to_dict(self) -> dict:
        return asdict(self)


# ── Question Templates ──────────────────────────────────────────────────

FIELD_QUESTIONS: Dict[str, str] = {
    "vendor_name": "What is the name of the vendor or company?",
    "invoice_number": "What is the invoice number?",
    "date": "What is the invoice date?",
    "total_amount": "What is the total amount due or paid?",
}


class NERService:
    """
    Extracts structured fields from invoice text using a QA transformer model.

    The model reads the raw OCR text as context and answers predefined questions
    to locate specific invoice fields (vendor, date, amount, invoice number).
    """

    def __init__(self, model_name: str = "deepset/roberta-base-squad2", confidence_threshold: float = 0.1):
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold

        logger.info("Loading NER model: %s", model_name)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
            self.model.eval()  # Set to evaluation mode (no dropout)
            logger.info("NER model loaded successfully")
        except Exception as exc:
            raise NERExtractionError(f"Failed to load NER model '{model_name}': {exc}")

    def extract_entities(self, text: str) -> NERResult:
        """
        Extract key invoice fields from raw text.

        Args:
            text: Raw text extracted from the invoice (e.g., via OCR).

        Returns:
            NERResult with vendor_name, invoice_number, date, total_amount.

        Raises:
            NERExtractionError: If extraction fails entirely.
        """
        if not text or not text.strip():
            raise NERExtractionError("Cannot extract entities from empty text")

        result = NERResult()

        for field_name, question in FIELD_QUESTIONS.items():
            extracted = self._answer_question(question, text)
            setattr(result, field_name, extracted)

        return result

    def extract_entities_as_json(self, text: str) -> str:
        """Extract entities and return as formatted JSON string (backward compat)."""
        result = self.extract_entities(text)
        output = {
            "status": "success",
            "entities": result.to_dict(),
        }
        return json.dumps(output, indent=4)

    def _answer_question(self, question: str, context: str) -> ExtractedField:
        """
        Ask a single question against the invoice text context.

        Uses the QA model to find the answer span, then computes a confidence
        score from the start/end position softmax probabilities.
        """
        try:
            inputs = self.tokenizer(
                question, context,
                return_tensors="pt",
                max_length=512,
                truncation=True,
            )

            with torch.no_grad():
                outputs = self.model(**inputs)

            # Find the most likely answer span
            start_idx = torch.argmax(outputs.start_logits)
            end_idx = torch.argmax(outputs.end_logits) + 1

            # Compute confidence as average of start/end softmax probabilities
            start_prob = torch.max(torch.softmax(outputs.start_logits, dim=-1)).item()
            end_prob = torch.max(torch.softmax(outputs.end_logits, dim=-1)).item()
            confidence = (start_prob + end_prob) / 2.0

            # Decode the answer tokens back to text
            answer_raw = self.tokenizer.decode(
                inputs.input_ids[0][start_idx:end_idx],
                skip_special_tokens=True,
            )
            answer = " ".join(answer_raw) if isinstance(answer_raw, list) else answer_raw
            answer = answer.strip()

            if confidence >= self.confidence_threshold and answer:
                return ExtractedField(value=answer, confidence=round(confidence, 4))
            else:
                return ExtractedField(
                    value=None,
                    confidence=round(confidence, 4),
                    note="Confidence below threshold or entity not found",
                )

        except Exception as exc:
            logger.error("Failed to extract field for question '%s': %s", question, exc)
            return ExtractedField(value=None, confidence=0.0, note=str(exc))
