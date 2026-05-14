"""Shared test fixtures for the Invoice Intelligence test suite."""

import json
import pytest
import numpy as np
from pathlib import Path


@pytest.fixture
def sample_invoices():
    """Load the sample invoice dataset."""
    data_path = Path(__file__).parent.parent / "data" / "sample_invoices.json"
    with open(data_path) as f:
        return json.load(f)


@pytest.fixture
def sample_embeddings():
    """Generate synthetic embeddings for testing."""
    np.random.seed(42)
    # 3 clusters + 2 outliers
    cluster_a = np.random.normal(loc=0.5, scale=0.05, size=(5, 384))
    cluster_b = np.random.normal(loc=-0.5, scale=0.05, size=(5, 384))
    cluster_c = np.random.normal(loc=0.0, scale=0.05, size=(3, 384))
    outliers = np.random.uniform(low=-2.0, high=2.0, size=(2, 384))
    return np.vstack([cluster_a, cluster_b, cluster_c, outliers])


@pytest.fixture
def sample_amounts():
    """Sample invoice amounts with known outliers."""
    return [500.0, 510.5, 495.0, 505.0, 490.0, 502.0, 498.0, 515.0,
            50000.0, 5.0, np.nan]
