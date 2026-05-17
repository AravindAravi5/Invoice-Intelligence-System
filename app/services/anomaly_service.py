"""
Anomaly Detection Service — Isolation Forest
==============================================
Identifies unusual or suspicious invoices using the Isolation Forest algorithm.

How Isolation Forest Works:
──────────────────────────
Unlike distance-based anomaly detectors (LOF, KNN), Isolation Forest takes
a fundamentally different approach: it explicitly *isolates* anomalies
rather than profiling what "normal" looks like.

Algorithm:
    1. Randomly select a feature (dimension of the embedding).
    2. Randomly select a split value between the min and max of that feature.
    3. Recursively partition the data with random splits.
    4. Anomalies are "few and different" — they get isolated in fewer splits
       (shorter path length in the isolation tree).
    5. Average the path length across many random trees (ensemble of 100 trees).

Why Isolation Forest for Invoice AI:
    ✓ Handles high-dimensional data (384-dim embeddings) efficiently
    ✓ Works well even with very few anomalies (< 1% of the dataset)
    ✓ Does NOT assume data follows a Gaussian distribution
    ✓ Linear time complexity: O(n × t × log(ψ)) where t = trees, ψ = sub-sample size
    ✓ No need for labeled anomaly data (fully unsupervised)

Key Parameter — contamination:
    The expected proportion of outliers in the dataset. This sets the
    decision threshold on the anomaly scores. For financial documents,
    typical values are 0.05–0.15 (5%–15% anomaly rate).
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Union, List

import numpy as np
from sklearn.ensemble import IsolationForest  # type: ignore
from sklearn.impute import SimpleImputer  # type: ignore

from app.exceptions import AnomalyDetectionError

logger = logging.getLogger(__name__)


@dataclass
class AnomalyResult:
    """Complete result from Isolation Forest anomaly detection."""
    labels: np.ndarray                      # 1 = normal, -1 = anomaly
    scores: np.ndarray                      # Raw anomaly scores (more negative = more anomalous)
    n_anomalies: int = 0
    n_normal: int = 0
    anomaly_percentage: float = 0.0
    score_threshold: Optional[float] = None  # The decision boundary score
    score_statistics: Dict[str, float] = field(default_factory=dict)
    parameters_used: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> Dict[str, Any]:
        """Human-readable summary for API responses."""
        return {
            "n_anomalies": self.n_anomalies,
            "n_normal": self.n_normal,
            "anomaly_percentage": round(self.anomaly_percentage, 2),
            "score_threshold": round(self.score_threshold, 4) if self.score_threshold else None,
            "score_statistics": {k: round(v, 4) for k, v in self.score_statistics.items()},
            "parameters": self.parameters_used,
        }

    def is_anomaly(self, index: int) -> bool:
        """Check if a specific sample is an anomaly."""
        return bool(self.labels[index] == -1)


class AnomalyService:
    """
    Isolation Forest-based anomaly detection for financial documents.

    Usage:
        service = AnomalyService(contamination=0.1)
        result = service.fit_predict(embeddings)
        print(result.summary())
    """

    def __init__(
        self,
        contamination: float = 0.1,
        n_estimators: int = 100,
        random_state: int = 42,
    ):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state

        # Imputer to handle NaN values (e.g., from failed OCR extraction)
        self._imputer = SimpleImputer(strategy="median")

        logger.info(
            "AnomalyService initialized — contamination=%.2f, n_estimators=%d",
            contamination, n_estimators,
        )

    def fit_predict(
        self,
        data: Union[np.ndarray, List[float]],
        is_1d_amounts: bool = False,
    ) -> AnomalyResult:
        """
        Detect anomalies in the provided data.

        Args:
            data: 2D numpy array of embeddings OR 1D list of invoice amounts.
            is_1d_amounts: If True, treats input as a flat list of amounts.

        Returns:
            AnomalyResult with labels, scores, and statistical analysis.
        """
        if data is None or len(data) == 0:
            raise AnomalyDetectionError("Cannot detect anomalies in empty data")

        X = np.array(data, dtype=float)

        # Reshape 1D data (e.g., invoice amounts) to 2D for sklearn
        if is_1d_amounts or len(X.shape) == 1:
            X = X.reshape(-1, 1)

        n_samples = len(X)
        logger.info("Analyzing %d records for anomalies...", n_samples)

        # ── Edge case: Too few samples ───────────────────────────────
        if n_samples < 5:
            logger.warning(
                "Dataset too small (%d samples) for reliable anomaly detection. "
                "Marking all as normal.",
                n_samples,
            )
            return AnomalyResult(
                labels=np.ones(n_samples, dtype=int),
                scores=np.zeros(n_samples),
                n_normal=n_samples,
                parameters_used=self._get_params(),
            )

        # ── Handle NaN values ────────────────────────────────────────
        if np.isnan(X).any():
            nan_count = int(np.isnan(X).sum())
            logger.warning(
                "%d NaN value(s) detected — imputing with median", nan_count
            )
            X = self._imputer.fit_transform(X)

        # ── Fit Isolation Forest ─────────────────────────────────────
        model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
        )
        labels = model.fit_predict(X)
        scores = model.score_samples(X)

        # ── Compute statistics ───────────────────────────────────────
        n_anomalies = int(np.sum(labels == -1))
        n_normal = int(np.sum(labels == 1))

        # Find the decision boundary (threshold score)
        score_threshold = float(model.offset_)

        score_stats = {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "min": float(np.min(scores)),
            "max": float(np.max(scores)),
            "median": float(np.median(scores)),
            "p5": float(np.percentile(scores, 5)),
            "p95": float(np.percentile(scores, 95)),
        }

        result = AnomalyResult(
            labels=labels,
            scores=scores,
            n_anomalies=n_anomalies,
            n_normal=n_normal,
            anomaly_percentage=(n_anomalies / n_samples) * 100,
            score_threshold=score_threshold,
            score_statistics=score_stats,
            parameters_used=self._get_params(),
        )

        logger.info(
            "Anomaly detection complete — %d anomalies found (%.1f%%)",
            n_anomalies, result.anomaly_percentage,
        )

        return result

    def fit_predict_with_contamination(
        self,
        data: np.ndarray,
        contamination: float,
    ) -> AnomalyResult:
        """
        Run anomaly detection with a custom contamination rate
        (for the Algorithm Explorer UI — interactive threshold tuning).
        """
        temp_service = AnomalyService(
            contamination=contamination,
            n_estimators=self.n_estimators,
            random_state=self.random_state,
        )
        return temp_service.fit_predict(data)

    def _get_params(self) -> Dict[str, Any]:
        return {
            "contamination": self.contamination,
            "n_estimators": self.n_estimators,
            "random_state": self.random_state,
        }
