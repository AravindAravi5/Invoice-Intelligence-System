"""
ML Services Layer
==================
Each service encapsulates one stage of the unsupervised ML pipeline:

- OCRService        : Text extraction from PDF/images
- NERService        : Named Entity Recognition (question-answering approach)
- EmbeddingService  : Sentence-Transformer dense vector embeddings
- ClusteringService : DBSCAN density-based clustering
- AnomalyService    : Isolation Forest anomaly detection
- PipelineService   : End-to-end orchestration of all stages
"""
