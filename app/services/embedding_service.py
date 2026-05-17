"""
Embedding Service — Sentence-Transformer Dense Vector Representations
=======================================================================
Converts structured invoice data into dense numerical vectors (embeddings)
that capture semantic meaning, enabling downstream unsupervised ML tasks.

Why Sentence-Transformers?
    Standard word embeddings (Word2Vec, GloVe) produce per-token vectors.
    Sentence-Transformers produce a single fixed-size vector for an entire
    text passage, optimized so that semantically similar texts have high
    cosine similarity. This is critical for DBSCAN (which uses cosine distance).

Model: all-MiniLM-L6-v2
    - Output dimension: 384
    - Fast inference (~14k sentences/sec on GPU)
    - Excellent quality-to-speed ratio

Dimensionality Reduction:
    For visualization, we provide t-SNE and PCA projections to map the
    384-dimensional embeddings down to 2D for scatter plots.
"""

import json
import logging
from typing import Dict, Any, List, Optional, Union

import numpy as np
from sklearn.decomposition import PCA  # type: ignore
from sklearn.manifold import TSNE  # type: ignore
from sklearn.metrics.pairwise import cosine_similarity  # type: ignore
from sentence_transformers import SentenceTransformer  # type: ignore

from app.exceptions import EmbeddingError

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Generates dense vector embeddings from invoice data for use in
    clustering (DBSCAN) and anomaly detection (Isolation Forest).
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        logger.info("Loading embedding model: %s", model_name)
        try:
            self.model = SentenceTransformer(model_name)
            dim_fn = getattr(self.model, "get_embedding_dimension", None) or self.model.get_sentence_embedding_dimension
            self._embedding_dim = dim_fn()
            logger.info(
                "Embedding model loaded — output dimension: %d", self._embedding_dim
            )
        except Exception as exc:
            raise EmbeddingError(f"Failed to load embedding model '{model_name}': {exc}")

    @property
    def embedding_dim(self) -> int:
        """Dimensionality of the output embedding vectors."""
        return int(self._embedding_dim)

    # ── Core Embedding Methods ───────────────────────────────────────

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a batch of raw text strings.

        Args:
            texts: List of text strings (e.g., raw OCR output).

        Returns:
            numpy array of shape (n_texts, embedding_dim).
        """
        if not texts:
            raise EmbeddingError("Cannot embed an empty text list")

        logger.info("Generating embeddings for %d text(s)...", len(texts))
        embeddings = self.model.encode(texts, show_progress_bar=False)
        return np.array(embeddings)

    def embed_invoice_entities(
        self, entities_batch: List[Dict[str, Any]]
    ) -> np.ndarray:
        """
        Generate embeddings from structured invoice entity dictionaries.

        Converts each entity dict into a natural language sentence before
        embedding, since sentence-transformers perform better on fluent
        text than on raw JSON syntax.

        Args:
            entities_batch: List of entity dicts (from NER extraction).

        Returns:
            numpy array of shape (n_invoices, embedding_dim).
        """
        texts = [self._entities_to_text(entities) for entities in entities_batch]
        valid = [t for t in texts if t.strip()]

        if not valid:
            raise EmbeddingError("No valid text extracted from entity batch")

        return self.embed_texts(valid)

    # ── Visualization Helpers (for the ML Visualizations tab) ────────

    def reduce_to_2d(
        self,
        embeddings: np.ndarray,
        method: str = "tsne",
        perplexity: float = 5.0,
        random_state: int = 42,
    ) -> np.ndarray:
        """
        Reduce high-dimensional embeddings to 2D for scatter plot visualization.

        Args:
            embeddings: Array of shape (n_samples, embedding_dim).
            method: 'tsne' or 'pca'.
            perplexity: t-SNE perplexity (typically 5–50). Lower for small datasets.
            random_state: Random seed for reproducibility.

        Returns:
            Array of shape (n_samples, 2).

        Why t-SNE?
            t-SNE preserves local neighborhood structure, making clusters visually
            apparent even when they overlap in high-dimensional space. PCA preserves
            global variance but may not separate clusters as clearly.
        """
        n_samples = embeddings.shape[0]

        if method == "tsne":
            # Adjust perplexity for small datasets (must be < n_samples)
            effective_perplexity = min(perplexity, max(2.0, n_samples - 1))
            reducer = TSNE(
                n_components=2,
                perplexity=effective_perplexity,
                random_state=random_state,
                max_iter=1000,
            )
        elif method == "pca":
            reducer = PCA(n_components=2, random_state=random_state)
        else:
            raise EmbeddingError(f"Unknown reduction method: {method}")

        logger.info("Reducing %d embeddings to 2D using %s", n_samples, method.upper())
        return np.array(reducer.fit_transform(embeddings))

    def compute_similarity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Compute pairwise cosine similarity matrix between all embeddings.

        Returns:
            Square matrix of shape (n_samples, n_samples) with values in [-1, 1].
            Values close to 1 indicate very similar invoices.
        """
        return np.array(cosine_similarity(embeddings))

    # ── Internal Helpers ─────────────────────────────────────────────

    @staticmethod
    def _entities_to_text(entities: Union[str, Dict[str, Any]]) -> str:
        """
        Convert structured entity data into a natural language string.

        Example:
            {"vendor_name": {"value": "Acme Corp"}} → "Vendor Name: Acme Corp."
        """
        if isinstance(entities, str):
            try:
                data = json.loads(entities)
            except json.JSONDecodeError:
                return entities.strip()
        else:
            data = entities

        # Handle nested NER output format
        if "entities" in data:
            data = data["entities"]

        parts = []
        for key, val in data.items():
            if isinstance(val, dict) and "value" in val:
                actual_value = val["value"]
            else:
                actual_value = val

            if actual_value is not None:
                clean_key = key.replace("_", " ").title()
                parts.append(f"{clean_key}: {actual_value}.")

        return " ".join(parts)
