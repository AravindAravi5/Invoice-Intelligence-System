"""
API Router — Invoice Processing Endpoints
===========================================
Defines all API routes for the Invoice Intelligence System.
"""

import os
import uuid
import json
import time
import logging
from pathlib import Path
from typing import List

from fastapi import APIRouter, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse

from app.api.schemas import (
    ProcessInvoiceResponse,
    BatchProcessResponse,
    HealthResponse,
    ErrorResponse,
)
from app.config import get_settings
from app.exceptions import InvoiceIntelligenceError, FileValidationError

logger = logging.getLogger(__name__)

router = APIRouter()
settings = get_settings()


@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check(request: Request):
    """
    Health check endpoint — reports system status and loaded models.
    Useful for Docker health checks and monitoring.
    """
    pipeline = request.app.state.pipeline
    start_time = getattr(request.app.state, "start_time", None)
    uptime = time.time() - start_time if start_time else None

    return HealthResponse(
        status="healthy",
        version=settings.app_version,
        models_loaded={
            "ocr": pipeline.ocr is not None,
            "ner": pipeline.ner is not None,
            "embeddings": pipeline.embedder is not None,
            "clustering": pipeline.clusterer is not None,
            "anomaly_detection": pipeline.anomaly_detector is not None,
        },
        uptime_seconds=round(uptime, 2) if uptime else None,
    )


@router.post("/process-invoice", tags=["Invoice Processing"])
async def process_invoice(request: Request, file: UploadFile = File(...)):
    """
    Process a single invoice through the full ML pipeline.

    Pipeline: Upload → OCR → NER → Embeddings → DBSCAN → Isolation Forest

    Accepts PDF, PNG, JPG, JPEG, and TIFF files.
    """
    # ── Validate file ────────────────────────────────────────────────
    filename = file.filename or "unknown.txt"
    ext = Path(filename).suffix.lower()
    if ext not in settings.allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{ext}'. Allowed: {settings.allowed_extensions}",
        )

    # ── Save to temp directory ───────────────────────────────────────
    os.makedirs(settings.upload_dir, exist_ok=True)
    file_path = os.path.join(settings.upload_dir, f"{uuid.uuid4()}_{filename}")

    try:
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)

        logger.info("Processing invoice: %s (%d bytes)", filename, len(content))

        # ── Run pipeline ─────────────────────────────────────────────
        pipeline = request.app.state.pipeline
        result = pipeline.process_single_file(file_path, filename=filename)

        return JSONResponse(content=result.to_api_response())

    except InvoiceIntelligenceError as exc:
        logger.error("Pipeline error at stage '%s': %s", exc.stage, exc.message)
        raise HTTPException(status_code=422, detail=str(exc.message))
    except Exception as exc:
        logger.exception("Unexpected error processing invoice")
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        # Always clean up temp file
        if os.path.exists(file_path):
            os.remove(file_path)


@router.post("/batch-process", tags=["Invoice Processing"])
async def batch_process(request: Request):
    """
    Process a batch of pre-extracted invoice entities through the ML pipeline.

    Skips OCR and NER — directly runs Embeddings → DBSCAN → Isolation Forest.
    Used for demo mode with sample data and re-running with different parameters.

    Request body: JSON array of invoice entity dictionaries.
    """
    try:
        body = await request.json()

        if not isinstance(body, list) or len(body) == 0:
            raise HTTPException(
                status_code=400,
                detail="Request body must be a non-empty JSON array of invoice entities",
            )

        pipeline = request.app.state.pipeline
        result = pipeline.process_batch_entities(body)

        return JSONResponse(content=result.to_api_response())

    except HTTPException:
        raise
    except InvoiceIntelligenceError as exc:
        raise HTTPException(status_code=422, detail=str(exc.message))
    except Exception as exc:
        logger.exception("Unexpected error in batch processing")
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/sample-data", tags=["Demo"])
async def get_sample_data():
    """
    Return the built-in sample invoice dataset for demo purposes.
    """
    sample_path = Path(__file__).parent.parent.parent / "data" / "sample_invoices.json"

    if not sample_path.exists():
        raise HTTPException(status_code=404, detail="Sample data file not found")

    with open(sample_path, "r") as f:
        data = json.load(f)

    return JSONResponse(content=data)
