"""
Clustering Service — DBSCAN Density-Based Clustering
======================================================
Groups similar invoices based on their dense vector embeddings using
DBSCAN (Density-Based Spatial Clustering of Applications with Noise).

Why DBSCAN for Invoice Clustering?
─────────────────────────────────
Unlike K-Means, DBSCAN offers three critical advantages for financial documents:

1. **No need to specify k** — We don't know how many vendors/categories exist
   in advance. DBSCAN discovers the natural number of clusters automatically.

2. **Noise detection** — Points that don't belong to any dense region are
   labeled as noise (-1). In financial contexts, noise points are often
   anomalous invoices worth investigating.

3. **Arbitrary cluster shapes** — K-Means assumes spherical clusters.
   DBSCAN can find clusters of any shape, which is important when invoice
   embeddings form irregular groupings in high-dimensional space.

Key Parameters:
    eps (ε): Maximum distance between two points to be considered neighbors.
             With cosine metric, eps=0.3 means cosine distance ≤ 0.3
             (i.e., cosine similarity ≥ 0.7).

    min_samples: Minimum points required to form a dense region (core point).
                 If a vendor has fewer invoices than min_samples, those
                 invoices will be classified as noise.

Evaluation:
    We use the Silhouette Score to evaluate clustering quality:
    - Score of +1: Clusters are dense and well-separated
    - Score of  0: Clusters overlap significantly
    - Score of -1: Points are assigned to wrong clusters
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, silhouette_samples

from app.exceptions import ClusteringError

logger = logging.getLogger(__name__)


@dataclass
class ClusteringResult:
    """Complete result from DBSCAN clustering analysis."""
    labels: np.ndarray                          # Cluster label per sample (-1 = noise)
    n_clusters: int = 0                         # Number of clusters found (excluding noise)
    n_noise: int = 0                            # Number of noise points
    noise_percentage: float = 0.0               # % of points classified as noise
    silhouette_avg: Optional[float] = None      # Average silhouette score (-1 to +1)
    silhouette_per_sample: Optional[np.ndarray] = None  # Per-sample silhouette scores
    cluster_sizes: Dict[int, int] = field(default_factory=dict)  # Cluster ID → size
    parameters_used: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> Dict[str, Any]:
        """Human-readable summary for API responses."""
        return {
            "n_clusters": self.n_clusters,
            "n_noise": self.n_noise,
            "noise_percentage": round(self.noise_percentage, 2),
            "silhouette_score": round(self.silhouette_avg, 4) if self.silhouette_avg is not None else None,
            "cluster_sizes": self.cluster_sizes,
            "parameters": self.parameters_used,
        }


class ClusteringService:
    """
    DBSCAN-based clustering for grouping similar invoices.

    Usage:
        service = ClusteringService(eps=0.3, min_samples=3)
        result = service.fit_predict(embeddings)
        print(result.summary())
    """

    def __init__(
        self,
        eps: float = 0.3,
        min_samples: int = 3,
        metric: str = "cosine",
    ):
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric

        logger.info(
            "ClusteringService initialized — eps=%.2f, min_samples=%d, metric='%s'",
            eps, min_samples, metric,
        )

    def fit_predict(self, embeddings: np.ndarray) -> ClusteringResult:
        """
        Cluster the provided embeddings using DBSCAN.

        Args:
            embeddings: 2D array of shape (n_samples, n_features).

        Returns:
            ClusteringResult with labels, statistics, and silhouette analysis.
        """
        if embeddings is None or len(embeddings) == 0:
            raise ClusteringError("Cannot cluster empty embeddings array")

        n_samples = len(embeddings)
        logger.info("Clustering %d invoice embeddings...", n_samples)

        # ── Handle small datasets ────────────────────────────────────
        effective_min_samples = self.min_samples
        if n_samples < self.min_samples:
            if n_samples >= 2:
                effective_min_samples = n_samples
                logger.warning(
                    "Dataset too small (%d samples) for min_samples=%d. "
                    "Temporarily reduced to %d.",
                    n_samples, self.min_samples, effective_min_samples,
                )
            else:
                return ClusteringResult(
                    labels=np.array([-1]),
                    n_noise=1,
                    noise_percentage=100.0,
                    parameters_used={"eps": self.eps, "min_samples": 1, "metric": self.metric},
                )

        # ── Run DBSCAN ───────────────────────────────────────────────
        model = DBSCAN(
            eps=self.eps,
            min_samples=effective_min_samples,
            metric=self.metric,
        )
        labels = model.fit_predict(embeddings)

        # ── Compute statistics ───────────────────────────────────────
        unique_labels = set(labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        n_noise = int(np.sum(labels == -1))

        cluster_sizes = {}
        for label in sorted(unique_labels):
            if label != -1:
                cluster_sizes[int(label)] = int(np.sum(labels == label))

        # ── Silhouette analysis ──────────────────────────────────────
        silhouette_avg = None
        silhouette_samples_arr = None

        if n_clusters >= 2 and n_noise < n_samples:
            try:
                # Only compute silhouette for non-noise points
                mask = labels != -1
                if mask.sum() >= 2:
                    silhouette_avg = float(silhouette_score(
                        embeddings[mask], labels[mask], metric=self.metric
                    ))
                    silhouette_samples_arr = silhouette_samples(
                        embeddings[mask], labels[mask], metric=self.metric
                    )
                    logger.info("Silhouette score: %.4f", silhouette_avg)
            except Exception as exc:
                logger.warning("Silhouette calculation failed: %s", exc)

        result = ClusteringResult(
            labels=labels,
            n_clusters=n_clusters,
            n_noise=n_noise,
            noise_percentage=(n_noise / n_samples) * 100,
            silhouette_avg=silhouette_avg,
            silhouette_per_sample=silhouette_samples_arr,
            cluster_sizes=cluster_sizes,
            parameters_used={
                "eps": self.eps,
                "min_samples": effective_min_samples,
                "metric": self.metric,
            },
        )

        logger.info(
            "Clustering complete — %d clusters found, %d noise points (%.1f%%)",
            n_clusters, n_noise, result.noise_percentage,
        )

        return result

    def fit_predict_with_params(
        self,
        embeddings: np.ndarray,
        eps: float,
        min_samples: int,
    ) -> ClusteringResult:
        """
        Run clustering with custom parameters (for the Algorithm Explorer UI).

        This allows interactive parameter tuning without modifying the service state.
        """
        temp_service = ClusteringService(
            eps=eps,
            min_samples=min_samples,
            metric=self.metric,
        )
        return temp_service.fit_predict(embeddings)
