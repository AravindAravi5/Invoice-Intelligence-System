# Invoice Intelligence System - Quick Start

## 🚀 Getting Started

### Option 1: Docker (Recommended)

**Windows:**
```bash
./deploy.bat
```

**Linux/macOS:**
```bash
bash deploy.sh
```

**Using Docker Compose:**
```bash
docker-compose up -d
```

### Option 2: Local Python Installation

```bash
# Create virtual environment
python -m venv .venv

# Activate
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/macOS

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```

## 📚 Documentation

- **Full Deployment Guide**: See [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)
- **API Documentation**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc

## 🧪 Test the API

### Using cURL
```bash
# Process an invoice
curl -F "file=@sample_invoice.pdf" http://localhost:8000/process-invoice
```

### Using Python
```python
import requests

with open('sample_invoice.pdf', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:8000/process-invoice', files=files)
    print(response.json())
```

### Using the API UI
Open http://localhost:8000/docs and use the interactive Swagger UI to test endpoints.

## 📋 Project Structure

```
invoice-system/
├── app.py                  # Main FastAPI application
├── ocr.py                  # OCR text extraction
├── ner.py                  # Named Entity Recognition
├── embeddings.py           # Text embeddings
├── clustering.py           # Invoice clustering
├── anomaly.py              # Anomaly detection
├── ui.py                   # (Optional UI components)
├── requirements.txt        # Python dependencies
├── Dockerfile              # Docker configuration
├── docker-compose.yml      # Multi-container setup
├── nginx.conf              # NGINX reverse proxy config
├── DEPLOYMENT_GUIDE.md     # Complete deployment guide
└── QUICKSTART.md           # This file
```

## 🔧 Configuration

Edit `.env` file to customize settings:

```bash
# Copy example config
cp .env.example .env

# Edit settings
nano .env
```

### Key Environment Variables
- `PORT`: API port (default: 8000)
- `LOG_LEVEL`: Logging level (INFO, DEBUG, etc.)
- `MAX_UPLOAD_SIZE_MB`: Max file size (default: 50)
- `CLUSTERING_EPS`: DBSCAN epsilon parameter
- `ANOMALY_CONTAMINATION`: Anomaly detection threshold

## 📊 Monitoring

```bash
# View logs
docker logs -f invoice-api

# Check container stats
docker stats invoice-api

# Access health endpoint
curl http://localhost:8000/health
```

## 🛑 Stopping

```bash
# Docker
docker stop invoice-api

# Docker Compose
docker-compose down

# Local (Ctrl+C in terminal)
```

## ⚠️ Troubleshooting

### Port 8000 already in use
```bash
# Find process using port 8000
lsof -i :8000  # macOS/Linux
netstat -ano | findstr :8000  # Windows

# Use different port in .env
PORT=8001
```

### OCR not working
- Ensure Tesseract is installed: `tesseract --version`
- For Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki

### Out of memory
- Reduce batch sizes in `.env`
- Increase container memory in `docker-compose.yml`
- Enable GPU support

## 📞 Support

Refer to [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) for comprehensive documentation and troubleshooting.

## 📜 License

[Add your license here]
