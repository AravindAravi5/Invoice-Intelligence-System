# 🧠 Invoice Intelligence System

> **AI-powered invoice processing pipeline using Unsupervised Machine Learning**
> 
> Automatically extracts, clusters, and detects anomalies in financial documents using DBSCAN, Isolation Forest, and Sentence-Transformer embeddings.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-FF4B4B?logo=streamlit)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikit-learn)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker)

---

## 📋 Table of Contents

- [Overview](#overview)
- [ML Pipeline Architecture](#ml-pipeline-architecture)
- [Unsupervised ML Algorithms](#unsupervised-ml-algorithms)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Demo Mode](#demo-mode)
- [API Documentation](#api-documentation)
- [Testing](#testing)
- [Docker Deployment](#docker-deployment)

---

## Overview

The Invoice Intelligence System processes financial documents through a **5-stage ML pipeline** that extracts text, identifies key entities, generates semantic embeddings, clusters similar invoices, and flags anomalies — all without labeled training data.

### Key Features

| Feature | Technology | Purpose |
|---------|-----------|---------|
| **OCR Text Extraction** | EasyOCR, Tesseract, PyMuPDF | Extract text from PDF/images |
| **Named Entity Recognition** | RoBERTa-SQuAD2 (QA approach) | Extract vendor, date, amount, invoice # |
| **Semantic Embeddings** | all-MiniLM-L6-v2 (384-dim) | Dense vector representation of invoices |
| **Density-Based Clustering** | DBSCAN (cosine metric) | Group similar invoices automatically |
| **Anomaly Detection** | Isolation Forest (100 trees) | Flag suspicious/unusual invoices |
| **Interactive Visualizations** | Plotly + Streamlit | t-SNE plots, heatmaps, parameter tuning |

---

## ML Pipeline Architecture

```
┌─────────────┐    ┌─────────────┐    ┌──────────────────┐    ┌─────────────┐    ┌──────────────────┐
│   📷 OCR    │───▶│  🏷️ NER    │───▶│ 🔢 Embeddings   │───▶│ 📍 DBSCAN  │───▶│ 🚨 Isolation    │
│             │    │             │    │                  │    │             │    │    Forest        │
│ PDF/Image   │    │ QA-based    │    │ Sentence-BERT    │    │ Density     │    │                  │
│ → Raw Text  │    │ extraction  │    │ 384-dim vectors  │    │ clustering  │    │ Anomaly scores   │
└─────────────┘    └─────────────┘    └──────────────────┘    └─────────────┘    └──────────────────┘
      ▼                   ▼                    ▼                     ▼                    ▼
  Raw text          JSON entities      numpy (n, 384)         Cluster labels       1=Normal, -1=Anomaly
```

**Data shape at each stage:**
1. **OCR** → `str` (raw text, ~500-2000 chars per invoice)
2. **NER** → `dict` with 4 fields (vendor, date, amount, invoice#) + confidence scores
3. **Embeddings** → `numpy.ndarray` shape `(n_invoices, 384)`
4. **DBSCAN** → `numpy.ndarray` shape `(n_invoices,)` — cluster labels (`-1` = noise)
5. **Isolation Forest** → `numpy.ndarray` shape `(n_invoices,)` — anomaly scores

---

## Unsupervised ML Algorithms

### DBSCAN (Density-Based Spatial Clustering)

**Why DBSCAN over K-Means?**
- ✅ No need to specify number of clusters `k` in advance
- ✅ Automatically identifies noise/outlier points (label = -1)
- ✅ Finds clusters of arbitrary shape
- ✅ Works well with cosine distance for high-dimensional embeddings

**Key Parameters:**
- `eps (ε)` — Maximum cosine distance between neighbors (default: 0.3)
- `min_samples` — Minimum points to form a cluster core (default: 3)

**Evaluation:** Silhouette Score (range: -1 to +1, higher = better separated clusters)

### Isolation Forest (Anomaly Detection)

**How it works:**
Instead of profiling "normal" data, Isolation Forest explicitly *isolates* anomalies through random recursive partitioning. Anomalous points are "few and different" — they get isolated in fewer splits (shorter path length).

**Key Parameters:**
- `contamination` — Expected proportion of anomalies (default: 0.1 = 10%)
- `n_estimators` — Number of isolation trees (default: 100)

**Why it's suitable:**
- ✅ Handles high-dimensional data (384-dim embeddings) efficiently
- ✅ Works well with very few anomalies (< 1% of dataset)
- ✅ No Gaussian distribution assumption
- ✅ Linear time complexity: O(n × t × log(ψ))

---

## Project Structure

```
invoice-intelligence-system/
├── app/                          # Backend application
│   ├── __init__.py
│   ├── main.py                   # FastAPI app factory with lifespan
│   ├── config.py                 # Centralized Pydantic settings
│   ├── exceptions.py             # Custom exception hierarchy
│   ├── api/
│   │   ├── router.py             # API endpoints
│   │   └── schemas.py            # Pydantic response models
│   └── services/
│       ├── ocr_service.py        # Multi-backend OCR (EasyOCR/Tesseract/PyMuPDF)
│       ├── ner_service.py        # QA-based entity extraction (RoBERTa)
│       ├── embedding_service.py  # Sentence-BERT + t-SNE/PCA reduction
│       ├── clustering_service.py # DBSCAN + silhouette analysis
│       ├── anomaly_service.py    # Isolation Forest + score statistics
│       └── pipeline.py           # End-to-end orchestrator with timing
├── ui/
│   └── streamlit_app.py          # 5-tab presentation-ready UI
├── tests/
│   ├── conftest.py               # Shared fixtures
│   ├── test_clustering.py        # DBSCAN tests (10 tests)
│   ├── test_anomaly.py           # Isolation Forest tests (11 tests)
│   └── test_embeddings.py        # Embedding tests (7 tests)
├── data/
│   └── sample_invoices.json      # 25 demo invoices (3 clusters + 4 anomalies)
├── Dockerfile                    # Multi-stage API build
├── Dockerfile.ui                 # Streamlit UI build
├── docker-compose.yml            # Full stack deployment
├── requirements.txt              # Python dependencies
├── pyproject.toml                # Project config + tool settings
├── .env.example                  # Environment variable template
└── README.md
```

---

## Quick Start

### Prerequisites
- Python 3.10+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/invoice-intelligence-system.git
cd invoice-intelligence-system

# Install dependencies
pip install -r requirements.txt
```

### Run the UI (Recommended for Demo)

```bash
python -m streamlit run ui/streamlit_app.py
```

Open `http://localhost:8501` → Click **"Load Sample Data & Run ML Pipeline"**

### Run the API Server

```bash
python -m uvicorn app.main:app --reload
```

API docs at `http://localhost:8000/docs`

---

## Demo Mode

The system includes **25 pre-built sample invoices** for instant demo (no OCR needed):

| Group | Vendor | Count | Amount Range | Purpose |
|-------|--------|-------|-------------|---------|
| Cluster A | Acme Consulting Group | 8 | $1,250–$2,100 | Consulting services |
| Cluster B | TechSupply Ltd | 8 | $275–$485 | Hardware supplies |
| Cluster C | CloudServ Inc | 5 | $79–$99 | SaaS subscriptions |
| Anomaly 1 | Unknown Offshore LLC | 1 | $95,000 | Suspiciously high amount |
| Anomaly 2 | Acme (mismatch) | 1 | $0.01 | Suspiciously low amount |
| Anomaly 3 | GlobalTrade Enterprises | 1 | $42,567 | Date from 1999 |
| Anomaly 4 | (missing fields) | 1 | $15,000 | All fields null |

---

## API Documentation

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | System health + model status |
| `POST` | `/process-invoice` | Process single invoice (PDF/image) |
| `POST` | `/batch-process` | Process batch of entity JSON |
| `GET` | `/sample-data` | Get built-in sample dataset |

### Example: Process Invoice

```bash
curl -X POST http://localhost:8000/process-invoice \
  -F "file=@invoice.pdf"
```

---

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=app --cov-report=term-missing
```

**Test Results: 21/21 passing** ✅

---

## Docker Deployment

```bash
# Build and run both API + UI
docker-compose up --build

# API: http://localhost:8000
# UI:  http://localhost:8501
```

---

## Technologies

| Category | Technology | Version |
|----------|-----------|---------|
| Backend | FastAPI + Uvicorn | 0.115+ |
| Frontend | Streamlit + Plotly | 1.30+ |
| NLP | Hugging Face Transformers (RoBERTa) | 4.30+ |
| Embeddings | Sentence-Transformers (MiniLM) | 2.2+ |
| ML | scikit-learn (DBSCAN, Isolation Forest) | 1.3+ |
| OCR | EasyOCR, Tesseract, PyMuPDF | Latest |
| Containerization | Docker + Docker Compose | Latest |

---

## License

MIT License — see [LICENSE](LICENSE) for details.
