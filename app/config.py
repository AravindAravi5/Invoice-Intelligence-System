"""
Centralized Configuration
==========================
Single source of truth for all application settings.
Uses Pydantic BaseSettings to load from environment variables with sensible defaults.

Usage:
    from app.config import get_settings
    settings = get_settings()
"""

import os
from functools import lru_cache
from typing import Optional, Set

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # ── Application ──────────────────────────────────────────────────────
    app_name: str = "Invoice Intelligence System"
    app_version: str = "2.0.0"
    debug: bool = False
    log_level: str = "INFO"

    # ── Server ───────────────────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000

    # ── File Upload ──────────────────────────────────────────────────────
    upload_dir: str = "temp_uploads"
    max_upload_size_mb: int = 50
    allowed_extensions: Set[str] = {".pdf", ".png", ".jpg", ".jpeg", ".tiff"}

    # ── OCR Dependencies ─────────────────────────────────────────────────
    tesseract_path: Optional[str] = None
    poppler_path: Optional[str] = None

    # ── ML Model Configuration ───────────────────────────────────────────
    # NER: Which Hugging Face QA model to use for entity extraction
    ner_model_name: str = "deepset/roberta-base-squad2"
    ner_confidence_threshold: float = 0.1

    # Embeddings: Sentence-transformer model for dense vector representations
    embedding_model_name: str = "all-MiniLM-L6-v2"

    # DBSCAN Clustering hyperparameters
    clustering_eps: float = 0.3
    clustering_min_samples: int = 3
    clustering_metric: str = "cosine"

    # Isolation Forest hyperparameters
    anomaly_contamination: float = 0.1
    anomaly_n_estimators: int = 100
    anomaly_random_state: int = 42

    # ── API ───────────────────────────────────────────────────────────────
    api_base_url: str = "http://localhost:8000"
    cors_origins: list = ["*"]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """
    Returns a cached singleton of the application settings.
    Using @lru_cache ensures the .env file is only read once.
    """
    return Settings()
