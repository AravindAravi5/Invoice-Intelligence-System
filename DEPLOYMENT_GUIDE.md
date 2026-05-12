# Invoice Intelligence System - Deployment Guide

## Overview
This guide covers deploying the Invoice & Financial Document Intelligence System across different environments.

---

## Prerequisites

### System Requirements
- **OS**: Linux, macOS, or Windows with WSL2
- **Python**: 3.11+
- **RAM**: Minimum 8GB (16GB recommended for ML models)
- **Disk Space**: 10GB+ (for model caches and uploads)
- **GPU** (Optional): NVIDIA GPU with CUDA support for faster processing

### Required System Dependencies
```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr poppler-utils libsm6 libxext6 libxrender-dev

# macOS
brew install tesseract poppler

# Windows (via Chocolatey)
choco install tesseract poppler
```

---

## Deployment Options

### Option 1: Local Development

#### Setup
```bash
# Clone/navigate to project directory
cd "Invoice & Financial Document Intelligence System"

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# (Optional) For GPU acceleration with PyTorch
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### Run the Application
```bash
python app.py
```

The API will be available at: **http://localhost:8000**
- API Documentation (Swagger UI): **http://localhost:8000/docs**
- Alternative API Docs (ReDoc): **http://localhost:8000/redoc**

---

### Option 2: Docker Deployment

#### Build Docker Image
```bash
docker build -t invoice-intelligence:latest .
```

#### Run Docker Container (Local)
```bash
docker run -p 8000:8000 \
  -v $(pwd)/temp_uploads:/app/temp_uploads \
  --name invoice-api \
  invoice-intelligence:latest
```

**Windows (PowerShell)**:
```powershell
docker run -p 8000:8000 `
  -v "${PWD}\temp_uploads:/app/temp_uploads" `
  --name invoice-api `
  invoice-intelligence:latest
```

#### Run with Docker Compose (Multi-service)
Create `docker-compose.yml`:
```yaml
version: '3.8'

services:
  invoice-api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./temp_uploads:/app/temp_uploads
      - ./logs:/app/logs
    environment:
      - LOG_LEVEL=INFO
      - HOST=0.0.0.0
      - PORT=8000
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/docs"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
```

Run:
```bash
docker-compose up -d
```

---

### Option 3: Cloud Deployment

#### AWS EC2
```bash
# 1. Launch EC2 instance (Ubuntu 22.04 LTS)
# 2. SSH into instance
ssh -i key.pem ubuntu@your-instance-ip

# 3. Install dependencies
sudo apt-get update
sudo apt-get install -y python3.11 python3.11-venv tesseract-ocr poppler-utils
sudo apt-get install -y libsm6 libxext6 libxrender-dev

# 4. Clone/upload project
git clone <your-repo> invoice-system
cd invoice-system

# 5. Setup and run
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 6. Run with PM2 or systemd (see below)
```

#### Google Cloud Run (Containerized)
```bash
# Build and push to Google Container Registry
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/invoice-api

# Deploy to Cloud Run
gcloud run deploy invoice-api \
  --image gcr.io/YOUR_PROJECT_ID/invoice-api \
  --platform managed \
  --region us-central1 \
  --memory 2Gi \
  --timeout 300
```

#### Azure Container Instances
```bash
az container create \
  --resource-group myResourceGroup \
  --name invoice-api \
  --image invoice-intelligence:latest \
  --ports 8000 \
  --environment-variables LOG_LEVEL=INFO PORT=8000
```

---

## Production Deployment

### Using PM2 (Process Manager)
```bash
# Install PM2
npm install -g pm2

# Create ecosystem.config.js
cat > ecosystem.config.js << 'EOF'
module.exports = {
  apps: [{
    name: 'invoice-api',
    script: './app.py',
    interpreter: '/path/to/.venv/bin/python',
    instances: 'max',
    exec_mode: 'cluster',
    env: {
      HOST: '0.0.0.0',
      PORT: 8000,
      LOG_LEVEL: 'INFO'
    },
    error_file: './logs/error.log',
    out_file: './logs/out.log',
    log_date_format: 'YYYY-MM-DD HH:mm:ss Z'
  }]
};
EOF

# Start application
pm2 start ecosystem.config.js

# Setup auto-restart on reboot
pm2 startup
pm2 save
```

### Using Systemd (Linux)
Create `/etc/systemd/system/invoice-api.service`:
```ini
[Unit]
Description=Invoice Intelligence API
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/home/ubuntu/invoice-system
Environment="PATH=/home/ubuntu/invoice-system/.venv/bin"
ExecStart=/home/ubuntu/invoice-system/.venv/bin/python app.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable invoice-api
sudo systemctl start invoice-api
sudo systemctl status invoice-api
```

---

## Reverse Proxy Setup

### NGINX Configuration
```nginx
upstream invoice_api {
    server 127.0.0.1:8000;
}

server {
    listen 80;
    server_name api.invoiceintelligence.com;

    client_max_body_size 50M;

    location / {
        proxy_pass http://invoice_api;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeouts for large file uploads
        proxy_connect_timeout 600s;
        proxy_send_timeout 600s;
        proxy_read_timeout 600s;
    }
}
```

### SSL/TLS with Let's Encrypt
```bash
sudo apt-get install certbot python3-certbot-nginx
sudo certbot certonly --nginx -d api.invoiceintelligence.com
```

---

## Monitoring & Logging

### Application Health Checks
```bash
# Check API status
curl http://localhost:8000/docs

# Monitor logs
tail -f logs/app.log

# Check resource usage
nvidia-smi  # GPU usage (if available)
```

### Logging Configuration
Logs are written to stdout and can be captured by Docker/systemd. For persistent logs, mount a volume or configure log aggregation (ELK Stack, Datadog, etc.).

---

## Performance Optimization

### Multi-Worker Setup
```bash
# Run with Gunicorn + multiple Uvicorn workers
pip install gunicorn
gunicorn app:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### GPU Acceleration
Ensure CUDA/cuDNN are installed and PyTorch is built with GPU support:
```python
# In app.py, verify GPU detection
import torch
print(f"GPU Available: {torch.cuda.is_available()}")
print(f"GPU Device: {torch.cuda.get_device_name(0)}")
```

### Caching
- Use Redis for model caching
- Implement request deduplication
- Cache OCR results for identical images

---

## Security Best Practices

1. **Environment Variables**: Use `.env` file (never commit secrets)
2. **File Uploads**: Validate file types, scan for malware
3. **Rate Limiting**: Implement API rate limiting (e.g., FastAPI-limiter)
4. **HTTPS**: Always use SSL/TLS in production
5. **Authentication**: Add API key/JWT authentication
6. **Input Validation**: Sanitize all user inputs
7. **Secrets Management**: Use AWS Secrets Manager, HashiCorp Vault, etc.

---

## Troubleshooting

### Model Loading Timeout
- Increase container memory/CPU
- Pre-warm models on startup
- Use model quantization for smaller sizes

### Out of Memory Errors
```bash
# Reduce batch size in .env
EMBEDDING_BATCH_SIZE=16

# Enable memory profiling
pip install memory-profiler
python -m memory_profiler app.py
```

### OCR Issues
- Ensure Tesseract is installed and in PATH
- For Windows: Install Tesseract-OCR from: https://github.com/UB-Mannheim/tesseract/wiki
- Adjust image preprocessing parameters in `ocr.py`

### Slow Processing
- Enable GPU acceleration
- Reduce model precision (float32 → float16)
- Implement asynchronous task queue (Celery + Redis)

---

## Scaling Strategies

### Horizontal Scaling
- Deploy multiple API instances behind a load balancer (NGINX, AWS ALB)
- Use containerization (Kubernetes for auto-scaling)
- Implement message queue (RabbitMQ, Redis) for async processing

### Vertical Scaling
- Increase CPU/RAM on the server
- Add GPU accelerators (NVIDIA GPUs)
- Optimize model architecture

---

## Maintenance

### Regular Updates
```bash
pip install --upgrade -r requirements.txt
```

### Backup Strategy
```bash
# Backup temp uploads and logs
tar -czf backup-$(date +%Y%m%d).tar.gz temp_uploads/ logs/
```

### Performance Monitoring
- Set up monitoring dashboards (Prometheus + Grafana)
- Track API response times, error rates, throughput
- Monitor system resources (CPU, RAM, Disk)

---

## Support & Documentation

- **API Docs**: http://your-server:8000/docs
- **Health Check**: GET `/health` (if implemented)
- **Logs**: Check application logs for errors

For issues, review the application logs and ensure all dependencies are properly installed.
