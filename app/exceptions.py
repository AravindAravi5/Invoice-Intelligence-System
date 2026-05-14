"""
Custom Exception Hierarchy
===========================
Structured exceptions for each pipeline stage.
Each maps to a specific HTTP status code and provides context for debugging.
"""


class InvoiceIntelligenceError(Exception):
    """Base exception for the Invoice Intelligence System."""

    def __init__(self, message: str, stage: str = "unknown", details: dict = None):
        self.message = message
        self.stage = stage
        self.details = details or {}
        super().__init__(self.message)


class FileValidationError(InvoiceIntelligenceError):
    """Raised when the uploaded file fails validation (format, size, corruption)."""

    def __init__(self, message: str, details: dict = None):
        super().__init__(message, stage="file_validation", details=details)


class OCRExtractionError(InvoiceIntelligenceError):
    """Raised when OCR text extraction fails."""

    def __init__(self, message: str, details: dict = None):
        super().__init__(message, stage="ocr", details=details)


class NERExtractionError(InvoiceIntelligenceError):
    """Raised when Named Entity Recognition fails to extract invoice fields."""

    def __init__(self, message: str, details: dict = None):
        super().__init__(message, stage="ner", details=details)


class EmbeddingError(InvoiceIntelligenceError):
    """Raised when sentence embedding generation fails."""

    def __init__(self, message: str, details: dict = None):
        super().__init__(message, stage="embedding", details=details)


class ClusteringError(InvoiceIntelligenceError):
    """Raised when DBSCAN clustering encounters an unrecoverable error."""

    def __init__(self, message: str, details: dict = None):
        super().__init__(message, stage="clustering", details=details)


class AnomalyDetectionError(InvoiceIntelligenceError):
    """Raised when Isolation Forest anomaly detection fails."""

    def __init__(self, message: str, details: dict = None):
        super().__init__(message, stage="anomaly_detection", details=details)


class PipelineError(InvoiceIntelligenceError):
    """Raised when the end-to-end pipeline orchestration fails."""

    def __init__(self, message: str, failed_stage: str = "unknown", details: dict = None):
        super().__init__(message, stage=f"pipeline.{failed_stage}", details=details)
