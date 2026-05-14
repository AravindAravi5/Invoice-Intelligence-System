"""
Pipeline Service — End-to-End ML Pipeline Orchestrator
========================================================
Coordinates all stages of the invoice processing pipeline:

    Upload → OCR → NER → Embeddings → DBSCAN Clustering → Isolation Forest

Each stage is timed and its results are captured for the API response
and UI visualization. The pipeline supports both single-file processing
and batch processing of pre-extracted invoice data.
"""

import json
import time
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from app.services.ocr_service import OCRService
from app.services.ner_service import NERService, NERResult
from app.services.embedding_service import EmbeddingService
from app.services.clustering_service import ClusteringService, ClusteringResult
from app.services.anomaly_service import AnomalyService, AnomalyResult
from app.exceptions import PipelineError

logger = logging.getLogger(__name__)


@dataclass
class StageTimings:
    """Execution time (seconds) for each pipeline stage."""
    ocr: float = 0.0
    ner: float = 0.0
    embedding: float = 0.0
    clustering: float = 0.0
    anomaly: float = 0.0
    total: float = 0.0


@dataclass
class SingleInvoiceResult:
    """Result of processing a single invoice through the pipeline."""
    filename: str
    raw_text: str
    text_length: int
    entities: Dict[str, Any]
    cluster_label: int
    is_anomaly: bool
    anomaly_score: float


@dataclass
class PipelineResult:
    """Complete result from the end-to-end pipeline execution."""
    invoices: List[SingleInvoiceResult]
    embeddings: Optional[np.ndarray] = None
    clustering: Optional[ClusteringResult] = None
    anomaly: Optional[AnomalyResult] = None
    timings: StageTimings = field(default_factory=StageTimings)

    def to_api_response(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict for API response."""
        invoice_dicts = []
        for inv in self.invoices:
            invoice_dicts.append({
                "filename": inv.filename,
                "text_snippet": inv.raw_text[:300] + "..." if len(inv.raw_text) > 300 else inv.raw_text,
                "text_length": inv.text_length,
                "entities": inv.entities,
                "cluster_label": inv.cluster_label,
                "is_anomaly": inv.is_anomaly,
                "anomaly_score": round(inv.anomaly_score, 4),
            })

        return {
            "status": "success",
            "total_processed": len(self.invoices),
            "invoices": invoice_dicts,
            "clustering_summary": self.clustering.summary() if self.clustering else None,
            "anomaly_summary": self.anomaly.summary() if self.anomaly else None,
            "embedding_shape": list(self.embeddings.shape) if self.embeddings is not None else None,
            "timings": asdict(self.timings),
        }


class PipelineService:
    """
    Orchestrates the full invoice processing pipeline.

    Holds references to all ML service instances and coordinates
    data flow between stages.
    """

    def __init__(
        self,
        ocr_service: OCRService,
        ner_service: NERService,
        embedding_service: EmbeddingService,
        clustering_service: ClusteringService,
        anomaly_service: AnomalyService,
    ):
        self.ocr = ocr_service
        self.ner = ner_service
        self.embedder = embedding_service
        self.clusterer = clustering_service
        self.anomaly_detector = anomaly_service
        logger.info("PipelineService initialized with all ML services")

    def process_single_file(self, file_path: str, filename: str = "") -> PipelineResult:
        """
        Process a single invoice file through the complete pipeline.

        Args:
            file_path: Path to the invoice file (PDF or image).
            filename: Original filename for the response.

        Returns:
            PipelineResult with all stage outputs.
        """
        if not filename:
            filename = Path(file_path).name

        timings = StageTimings()
        pipeline_start = time.time()

        try:
            # ── Stage 1: OCR ─────────────────────────────────────────
            logger.info("[Pipeline] Stage 1/5: OCR extraction")
            t0 = time.time()
            raw_text = self.ocr.extract_text(file_path)
            timings.ocr = time.time() - t0

            if not raw_text.strip():
                raise PipelineError("OCR returned empty text", failed_stage="ocr")

            # ── Stage 2: NER ─────────────────────────────────────────
            logger.info("[Pipeline] Stage 2/5: Entity extraction")
            t0 = time.time()
            ner_result = self.ner.extract_entities(raw_text)
            entities_dict = ner_result.to_dict()
            timings.ner = time.time() - t0

            # ── Stage 3: Embeddings ──────────────────────────────────
            logger.info("[Pipeline] Stage 3/5: Generating embeddings")
            t0 = time.time()
            embeddings = self.embedder.embed_invoice_entities([entities_dict])
            timings.embedding = time.time() - t0

            # ── Stage 4: Clustering ──────────────────────────────────
            logger.info("[Pipeline] Stage 4/5: DBSCAN clustering")
            t0 = time.time()
            clustering_result = self.clusterer.fit_predict(embeddings)
            timings.clustering = time.time() - t0

            # ── Stage 5: Anomaly Detection ───────────────────────────
            logger.info("[Pipeline] Stage 5/5: Isolation Forest anomaly detection")
            t0 = time.time()
            anomaly_result = self.anomaly_detector.fit_predict(embeddings)
            timings.anomaly = time.time() - t0

            timings.total = time.time() - pipeline_start

            invoice_result = SingleInvoiceResult(
                filename=filename,
                raw_text=raw_text,
                text_length=len(raw_text),
                entities=entities_dict,
                cluster_label=int(clustering_result.labels[0]),
                is_anomaly=bool(anomaly_result.labels[0] == -1),
                anomaly_score=float(anomaly_result.scores[0]),
            )

            logger.info(
                "[Pipeline] Complete — %.2fs total (OCR: %.2fs, NER: %.2fs, "
                "Embed: %.2fs, Cluster: %.2fs, Anomaly: %.2fs)",
                timings.total, timings.ocr, timings.ner,
                timings.embedding, timings.clustering, timings.anomaly,
            )

            return PipelineResult(
                invoices=[invoice_result],
                embeddings=embeddings,
                clustering=clustering_result,
                anomaly=anomaly_result,
                timings=timings,
            )

        except PipelineError:
            raise
        except Exception as exc:
            raise PipelineError(f"Pipeline failed: {exc}", failed_stage="unknown")

    def process_batch_entities(
        self,
        entities_batch: List[Dict[str, Any]],
        filenames: Optional[List[str]] = None,
    ) -> PipelineResult:
        """
        Process a batch of pre-extracted invoice entities (skips OCR + NER).

        This is used for:
        - Demo mode with sample_invoices.json
        - Re-running clustering/anomaly detection with different parameters
        - ML Visualizations tab

        Args:
            entities_batch: List of entity dictionaries.
            filenames: Optional list of filenames for display.

        Returns:
            PipelineResult with embeddings, clustering, and anomaly detection.
        """
        n = len(entities_batch)
        if filenames is None:
            filenames = [f"invoice_{i+1}" for i in range(n)]

        timings = StageTimings()
        pipeline_start = time.time()

        try:
            # ── Stage 3: Embeddings ──────────────────────────────────
            logger.info("[Batch Pipeline] Generating embeddings for %d invoices", n)
            t0 = time.time()
            embeddings = self.embedder.embed_invoice_entities(entities_batch)
            timings.embedding = time.time() - t0

            # ── Stage 4: Clustering ──────────────────────────────────
            logger.info("[Batch Pipeline] Running DBSCAN clustering")
            t0 = time.time()
            clustering_result = self.clusterer.fit_predict(embeddings)
            timings.clustering = time.time() - t0

            # ── Stage 5: Anomaly Detection ───────────────────────────
            logger.info("[Batch Pipeline] Running Isolation Forest")
            t0 = time.time()
            anomaly_result = self.anomaly_detector.fit_predict(embeddings)
            timings.anomaly = time.time() - t0

            timings.total = time.time() - pipeline_start

            # ── Build per-invoice results ────────────────────────────
            invoices = []
            for i in range(n):
                entity_text = json.dumps(entities_batch[i], default=str)
                invoices.append(SingleInvoiceResult(
                    filename=filenames[i],
                    raw_text=entity_text,
                    text_length=len(entity_text),
                    entities=entities_batch[i],
                    cluster_label=int(clustering_result.labels[i]),
                    is_anomaly=bool(anomaly_result.labels[i] == -1),
                    anomaly_score=float(anomaly_result.scores[i]),
                ))

            logger.info(
                "[Batch Pipeline] Complete — %d invoices in %.2fs "
                "(Embed: %.2fs, Cluster: %.2fs, Anomaly: %.2fs)",
                n, timings.total, timings.embedding,
                timings.clustering, timings.anomaly,
            )

            return PipelineResult(
                invoices=invoices,
                embeddings=embeddings,
                clustering=clustering_result,
                anomaly=anomaly_result,
                timings=timings,
            )

        except PipelineError:
            raise
        except Exception as exc:
            raise PipelineError(f"Batch pipeline failed: {exc}", failed_stage="unknown")
