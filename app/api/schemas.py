"""
API Response Schemas
=====================
Pydantic models for typed, validated API responses.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ExtractedFieldSchema(BaseModel):
    """A single NER-extracted invoice field."""
    value: Optional[str] = None
    confidence: float = 0.0
    note: Optional[str] = None


class InvoiceEntitiesSchema(BaseModel):
    """Structured fields extracted from an invoice."""
    vendor_name: ExtractedFieldSchema = Field(default_factory=ExtractedFieldSchema)
    invoice_number: ExtractedFieldSchema = Field(default_factory=ExtractedFieldSchema)
    date: ExtractedFieldSchema = Field(default_factory=ExtractedFieldSchema)
    total_amount: ExtractedFieldSchema = Field(default_factory=ExtractedFieldSchema)


class OCRResultSchema(BaseModel):
    """OCR extraction result."""
    text_snippet: str = ""
    text_length: int = 0


class ClusteringResultSchema(BaseModel):
    """DBSCAN clustering result for a single invoice."""
    cluster_label: int = -1
    description: str = "Unclustered"


class AnomalyResultSchema(BaseModel):
    """Isolation Forest result for a single invoice."""
    is_anomaly: bool = False
    anomaly_score: float = 0.0
    description: str = "Normal"


class MLResultSchema(BaseModel):
    """Combined ML analysis results."""
    embedding_shape: List[int] = []
    clustering: ClusteringResultSchema = Field(default_factory=ClusteringResultSchema)
    anomaly_detection: AnomalyResultSchema = Field(default_factory=AnomalyResultSchema)


class InvoiceResultSchema(BaseModel):
    """Full processing result for a single invoice."""
    filename: str
    ocr: OCRResultSchema
    entities: Dict[str, Any]
    ml: MLResultSchema


class ProcessInvoiceResponse(BaseModel):
    """Response from the /process-invoice endpoint."""
    status: str = "success"
    total_processed: int = 1
    invoices: List[InvoiceResultSchema]
    clustering_summary: Optional[Dict[str, Any]] = None
    anomaly_summary: Optional[Dict[str, Any]] = None
    timings: Optional[Dict[str, float]] = None


class BatchProcessResponse(BaseModel):
    """Response from the /batch-process endpoint."""
    status: str = "success"
    total_processed: int
    invoices: List[Dict[str, Any]]
    clustering_summary: Optional[Dict[str, Any]] = None
    anomaly_summary: Optional[Dict[str, Any]] = None
    embedding_shape: Optional[List[int]] = None
    timings: Optional[Dict[str, float]] = None


class HealthResponse(BaseModel):
    """Response from the /health endpoint."""
    status: str = "healthy"
    version: str
    models_loaded: Dict[str, bool]
    uptime_seconds: Optional[float] = None


class ErrorResponse(BaseModel):
    """Structured error response."""
    status: str = "error"
    stage: str = "unknown"
    message: str
    details: Optional[Dict[str, Any]] = None
