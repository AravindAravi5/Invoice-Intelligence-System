"""Tests for Isolation Forest Anomaly Detection Service."""

import numpy as np
import pytest
from app.services.anomaly_service import AnomalyService


class TestAnomalyService:
    """Test suite for the Isolation Forest anomaly detection module."""

    def test_basic_detection(self, sample_embeddings):
        """Should detect anomalies in data with known outliers."""
        service = AnomalyService(contamination=0.15)
        result = service.fit_predict(sample_embeddings)
        assert result.n_anomalies > 0, "Should detect at least 1 anomaly"
        assert result.n_normal > 0, "Should have normal points too"

    def test_labels_shape(self, sample_embeddings):
        """Labels should match input size."""
        service = AnomalyService()
        result = service.fit_predict(sample_embeddings)
        assert len(result.labels) == len(sample_embeddings)
        assert len(result.scores) == len(sample_embeddings)

    def test_label_values(self, sample_embeddings):
        """Labels should be 1 (normal) or -1 (anomaly)."""
        service = AnomalyService()
        result = service.fit_predict(sample_embeddings)
        unique = set(result.labels)
        assert unique.issubset({1, -1})

    def test_1d_amounts(self, sample_amounts):
        """Should work with 1D invoice amounts."""
        service = AnomalyService(contamination=0.2)
        result = service.fit_predict(sample_amounts, is_1d_amounts=True)
        assert len(result.labels) == len(sample_amounts)

    def test_nan_handling(self):
        """Should handle NaN values gracefully."""
        service = AnomalyService()
        data = np.array([1.0, 2.0, np.nan, 3.0, 4.0, 5.0, 6.0, 100.0])
        result = service.fit_predict(data, is_1d_amounts=True)
        assert len(result.labels) == 8

    def test_small_dataset(self):
        """Should mark all as normal for very small datasets."""
        service = AnomalyService()
        data = np.random.rand(3, 10)
        result = service.fit_predict(data)
        assert result.n_anomalies == 0
        assert np.all(result.labels == 1)

    def test_score_statistics(self, sample_embeddings):
        """Score statistics should contain expected keys."""
        service = AnomalyService()
        result = service.fit_predict(sample_embeddings)
        stats = result.score_statistics
        assert "mean" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats

    def test_is_anomaly_method(self, sample_embeddings):
        """is_anomaly helper should work correctly."""
        service = AnomalyService(contamination=0.15)
        result = service.fit_predict(sample_embeddings)
        for i in range(len(sample_embeddings)):
            assert result.is_anomaly(i) == (result.labels[i] == -1)

    def test_summary(self, sample_embeddings):
        """Summary should be a valid dict."""
        service = AnomalyService()
        result = service.fit_predict(sample_embeddings)
        summary = result.summary()
        assert "n_anomalies" in summary
        assert "anomaly_percentage" in summary

    def test_empty_input_raises(self):
        """Should raise on empty input."""
        service = AnomalyService()
        with pytest.raises(Exception):
            service.fit_predict(np.array([]))

    def test_contamination_tuning(self, sample_embeddings):
        """Different contamination rates should yield different anomaly counts."""
        service = AnomalyService()
        r1 = service.fit_predict_with_contamination(sample_embeddings, 0.05)
        r2 = service.fit_predict_with_contamination(sample_embeddings, 0.3)
        assert r2.n_anomalies >= r1.n_anomalies
