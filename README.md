# Invoice & Financial Document Intelligence System

An AI-driven system for processing, analyzing, and extracting insights from invoices and financial documents.

## Features
- **OCR (Optical Character Recognition)**: Extract text from images and PDFs using EasyOCR and PyMuPDF.
- **NER (Named Entity Recognition)**: Identify key entities like Vendor, Date, and Amount using Hugging Face Transformers.
- **Anomaly Detection**: Detect unusual patterns or fraudulent entries in financial data.
- **Clustering & Categorization**: Group invoices based on content similarity.
- **Interactive UI**: User-friendly interface built with Streamlit.

## Quick Start

### Prerequisites
- Python 3.8+
- Git

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/invoice-intelligence-system.git
   cd invoice-intelligence-system
   ```
2. Install dependencies:
   ```bash
   install-dependencies.bat
   ```
   Or manually:
   ```bash
   pip install -r requirements.txt
   ```

### Running the App
```bash
streamlit run ui.py
```

## Documentation
- [QUICKSTART.md](QUICKSTART.md)
- [WINDOWS_SETUP.md](WINDOWS_SETUP.md)
- [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)

## License
MIT
