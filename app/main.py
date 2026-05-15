"""
FastAPI Application Factory
=============================
Creates and configures the FastAPI application with proper lifecycle management.

- ML models are loaded during startup via the lifespan context manager
- CORS is configured for cross-origin requests (Streamlit → FastAPI)
- Custom exception handlers return structured JSON errors
"""

import time
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app import __version__
from app.config import get_settings
from app.exceptions import InvoiceIntelligenceError
from app.api.router import router
from app.services.ocr_service import OCRService
from app.services.ner_service import NERService
from app.services.embedding_service import EmbeddingService
from app.services.clustering_service import ClusteringService
from app.services.anomaly_service import AnomalyService
from app.services.pipeline import PipelineService

logger = logging.getLogger(__name__)
settings = get_settings()

# ── Configure Logging ────────────────────────────────────────────────────

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


# ── Application Lifespan ─────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(fastapi_app: FastAPI):
    """
    Manages ML model lifecycle.

    Startup: Load all ML models into memory (OCR, NER, Embeddings, etc.)
    Shutdown: Clean up resources
    """
    logger.info("=" * 60)
    logger.info("  Invoice Intelligence System v%s — Starting Up", __version__)
    logger.info("=" * 60)

    fastapi_app.state.start_time = time.time()

    # ── Initialize ML Services ───────────────────────────────────────
    logger.info("Loading ML models (this may take a moment)...")

    try:
        ocr_service = OCRService(
            tesseract_cmd=settings.tesseract_path,
            poppler_path=settings.poppler_path,
        )

        ner_service = NERService(
            model_name=settings.ner_model_name,
            confidence_threshold=settings.ner_confidence_threshold,
        )

        embedding_service = EmbeddingService(
            model_name=settings.embedding_model_name,
        )

        clustering_service = ClusteringService(
            eps=settings.clustering_eps,
            min_samples=settings.clustering_min_samples,
            metric=settings.clustering_metric,
        )

        anomaly_service = AnomalyService(
            contamination=settings.anomaly_contamination,
            n_estimators=settings.anomaly_n_estimators,
            random_state=settings.anomaly_random_state,
        )

        # ── Assemble Pipeline ────────────────────────────────────────
        fastapi_app.state.pipeline = PipelineService(
            ocr_service=ocr_service,
            ner_service=ner_service,
            embedding_service=embedding_service,
            clustering_service=clustering_service,
            anomaly_service=anomaly_service,
        )

        # Store individual services for direct access (e.g., re-clustering)
        fastapi_app.state.embedding_service = embedding_service
        fastapi_app.state.clustering_service = clustering_service
        fastapi_app.state.anomaly_service = anomaly_service

        elapsed = time.time() - fastapi_app.state.start_time
        logger.info("All ML models loaded successfully in %.1fs", elapsed)
        logger.info("=" * 60)

    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.error("Failed to load ML models: %s", exc)
        logger.warning("Server will start but endpoints may fail")

    yield  # ← Application runs here

    # ── Shutdown ─────────────────────────────────────────────────────
    logger.info("Shutting down Invoice Intelligence System...")


# ── Create Application ───────────────────────────────────────────────────

def create_app() -> FastAPI:
    """Application factory — creates and configures the FastAPI instance."""
    app = FastAPI(
        title="Invoice Intelligence API",
        description=(
            "AI-powered backend for processing, analyzing, and clustering "
            "financial documents using Unsupervised Machine Learning.\n\n"
            "**Pipeline:** OCR → NER → Sentence Embeddings → DBSCAN Clustering → Isolation Forest"
        ),
        version=__version__,
        lifespan=lifespan,
    )

    # ── CORS Middleware ──────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Exception Handlers ───────────────────────────────────────────
    @app.exception_handler(InvoiceIntelligenceError)
    async def handle_pipeline_error(_request: Request, exc: InvoiceIntelligenceError):
        return JSONResponse(
            status_code=422,
            content={
                "status": "error",
                "stage": exc.stage,
                "message": exc.message,
                "details": exc.details,
            },
        )

    # ── Register Routes ──────────────────────────────────────────────
    app.include_router(router, prefix="/api/v1")

    # Also mount at root for backward compatibility
    app.include_router(router)

    return app


# ── Module-level app instance (for uvicorn) ──────────────────────────────
app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )
