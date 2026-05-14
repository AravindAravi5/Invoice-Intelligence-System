"""Tests for DBSCAN Clustering Service."""

import numpy as np
import pytest
from app.services.clustering_service import ClusteringService


class TestClusteringService:
    """Test suite for the DBSCAN clustering module."""

    def test_basic_clustering(self, sample_embeddings):
        """Should find clusters in well-separated data."""
        service = ClusteringService(eps=0.3, min_samples=2)
        result = service.fit_predict(sample_embeddings)
        assert result.n_clusters >= 2, "Should find at least 2 clusters"
        assert len(result.labels) == len(sample_embeddings)

    def test_cluster_labels_shape(self, sample_embeddings):
        """Labels array should match input size."""
        service = ClusteringService(eps=0.3, min_samples=2)
        result = service.fit_predict(sample_embeddings)
        assert result.labels.shape == (len(sample_embeddings),)

    def test_noise_detection(self, sample_embeddings):
        """Outliers should be labeled as noise (-1)."""
        service = ClusteringService(eps=0.1, min_samples=3)
        result = service.fit_predict(sample_embeddings)
        assert result.n_noise > 0, "Should detect noise points"
        assert -1 in result.labels

    def test_single_sample(self):
        """Single sample should be labeled as noise."""
        service = ClusteringService()
        data = np.random.rand(1, 384)
        result = service.fit_predict(data)
        assert result.labels[0] == -1

    def test_small_dataset(self):
        """Should handle dataset smaller than min_samples."""
        service = ClusteringService(min_samples=5)
        data = np.random.rand(3, 384)
        result = service.fit_predict(data)
        assert len(result.labels) == 3

    def test_silhouette_score(self, sample_embeddings):
        """Silhouette score should be computed for multi-cluster results."""
        service = ClusteringService(eps=0.3, min_samples=2)
        result = service.fit_predict(sample_embeddings)
        if result.n_clusters >= 2:
            assert result.silhouette_avg is not None
            assert -1 <= result.silhouette_avg <= 1

    def test_cluster_sizes(self, sample_embeddings):
        """Cluster sizes should sum to total non-noise points."""
        service = ClusteringService(eps=0.3, min_samples=2)
        result = service.fit_predict(sample_embeddings)
        total_clustered = sum(result.cluster_sizes.values())
        assert total_clustered == len(sample_embeddings) - result.n_noise

    def test_summary_output(self, sample_embeddings):
        """Summary should be a valid dict."""
        service = ClusteringService(eps=0.3, min_samples=2)
        result = service.fit_predict(sample_embeddings)
        summary = result.summary()
        assert "n_clusters" in summary
        assert "silhouette_score" in summary

    def test_empty_input_raises(self):
        """Should raise on empty input."""
        service = ClusteringService()
        with pytest.raises(Exception):
            service.fit_predict(np.array([]))

    def test_interactive_params(self, sample_embeddings):
        """fit_predict_with_params should work with custom params."""
        service = ClusteringService()
        r1 = service.fit_predict_with_params(sample_embeddings, eps=0.1, min_samples=2)
        r2 = service.fit_predict_with_params(sample_embeddings, eps=0.5, min_samples=2)
        # Different eps should potentially give different results
        assert len(r1.labels) == len(r2.labels)
