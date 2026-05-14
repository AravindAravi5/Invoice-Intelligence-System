"""Tests for Sentence-Transformer Embedding Service."""

import numpy as np
import pytest
from app.services.embedding_service import EmbeddingService


@pytest.fixture(scope="module")
def embedding_service():
    """Shared embedding service instance (model loaded once)."""
    return EmbeddingService()


class TestEmbeddingService:
    """Test suite for the embedding generation module."""

    def test_embed_texts(self, embedding_service):
        """Should generate embeddings for raw texts."""
        texts = ["Acme Corp Invoice $500", "TechSupply Ltd Invoice $300"]
        embeddings = embedding_service.embed_texts(texts)
        assert embeddings.shape[0] == 2
        assert embeddings.shape[1] > 0

    def test_embed_invoice_entities(self, embedding_service, sample_invoices):
        """Should generate embeddings from invoice entity dicts."""
        embeddings = embedding_service.embed_invoice_entities(sample_invoices[:5])
        assert embeddings.shape[0] == 5

    def test_similar_invoices_close(self, embedding_service):
        """Similar invoices should have high cosine similarity."""
        entities = [
            {"vendor_name": {"value": "Acme Corp"}, "total_amount": {"value": "$500"}},
            {"vendor_name": {"value": "Acme Corp"}, "total_amount": {"value": "$550"}},
            {"vendor_name": {"value": "Globex Inc"}, "total_amount": {"value": "$10000"}},
        ]
        emb = embedding_service.embed_invoice_entities(entities)
        sim = embedding_service.compute_similarity_matrix(emb)
        # Acme invoices should be more similar to each other than to Globex
        assert sim[0, 1] > sim[0, 2]

    def test_reduce_to_2d_tsne(self, embedding_service):
        """t-SNE should produce 2D output."""
        emb = np.random.rand(10, 384)
        coords = embedding_service.reduce_to_2d(emb, method="tsne")
        assert coords.shape == (10, 2)

    def test_reduce_to_2d_pca(self, embedding_service):
        """PCA should produce 2D output."""
        emb = np.random.rand(10, 384)
        coords = embedding_service.reduce_to_2d(emb, method="pca")
        assert coords.shape == (10, 2)

    def test_similarity_matrix_shape(self, embedding_service):
        """Similarity matrix should be square."""
        emb = np.random.rand(5, 384)
        sim = embedding_service.compute_similarity_matrix(emb)
        assert sim.shape == (5, 5)

    def test_empty_input_raises(self, embedding_service):
        """Should raise on empty input."""
        with pytest.raises(Exception):
            embedding_service.embed_texts([])
