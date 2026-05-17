"""
OCR Service — Optical Character Recognition
=============================================
Extracts raw text from invoice images and PDFs using a multi-backend approach:

1. **PyMuPDF (fitz)** — For native-text PDFs (fastest, most accurate)
2. **EasyOCR** — Deep learning based OCR (handles complex layouts well)
3. **Tesseract** — Traditional OCR engine (reliable fallback)

Preprocessing Pipeline:
    Raw Image → Grayscale → Contrast Enhancement → Sharpening → OCR Engine
"""

import os
import logging
from typing import Optional

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter  # type: ignore

from app.exceptions import OCRExtractionError

logger = logging.getLogger(__name__)


class OCRService:
    """
    Multi-backend OCR text extraction service.

    Supports PDF and image files with automatic backend selection
    and image preprocessing for improved accuracy.
    """

    SUPPORTED_IMAGES = {".png", ".jpg", ".jpeg", ".tiff", ".bmp"}
    SUPPORTED_FORMATS = SUPPORTED_IMAGES | {".pdf"}

    def __init__(
        self,
        tesseract_cmd: Optional[str] = None,
        poppler_path: Optional[str] = None,
    ):
        self.poppler_path = poppler_path
        self._tesseract_available = False
        self._easyocr_reader = None

        # ── Configure Tesseract ──────────────────────────────────────
        if tesseract_cmd:
            try:
                import pytesseract  # type: ignore

                pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
                self._tesseract_available = True
                logger.info("Tesseract OCR configured at: %s", tesseract_cmd)
            except ImportError:
                logger.warning("pytesseract not installed — Tesseract backend unavailable")
        else:
            try:
                import pytesseract  # type: ignore

                pytesseract.get_tesseract_version()
                self._tesseract_available = True
                logger.info("Tesseract OCR detected in system PATH")
            except Exception:
                logger.info("Tesseract not found in PATH — will use other backends")

        # ── Initialize EasyOCR ───────────────────────────────────────
        try:
            import easyocr  # type: ignore

            self._easyocr_reader = easyocr.Reader(["en"], verbose=False)
            logger.info("EasyOCR initialized successfully")
        except Exception as exc:
            logger.info("EasyOCR unavailable: %s", exc)

    # ── Public Interface ─────────────────────────────────────────────

    def extract_text(self, file_path: str) -> str:
        """
        Main entry point — extract text from any supported file.

        Args:
            file_path: Absolute or relative path to PDF or image file.

        Returns:
            Extracted text as a string.

        Raises:
            FileNotFoundError: If the file does not exist.
            OCRExtractionError: If text extraction fails.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()

        if ext not in self.SUPPORTED_FORMATS:
            raise OCRExtractionError(
                f"Unsupported format '{ext}'. Supported: {self.SUPPORTED_FORMATS}"
            )

        try:
            if ext == ".pdf":
                return self._extract_from_pdf(file_path)
            else:
                with Image.open(file_path) as img:
                    return self._extract_from_image(img)
        except OCRExtractionError:
            raise
        except Exception as exc:
            raise OCRExtractionError(f"Extraction failed for {file_path}: {exc}")

    # ── Image Processing ─────────────────────────────────────────────

    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """
        Preprocessing pipeline to improve OCR accuracy on noisy/low-quality scans.

        Steps:
            1. Convert to grayscale — removes color noise
            2. Contrast enhancement (2×) — makes text stand out
            3. Sharpening filter — clarifies blurry edges
        """
        try:
            image = image.convert("L")
            image = ImageEnhance.Contrast(image).enhance(2.0)
            image = image.filter(ImageFilter.SHARPEN)
            return image
        except Exception as exc:
            raise OCRExtractionError(f"Image preprocessing failed: {exc}")

    def _extract_from_image(self, image: Image.Image) -> str:
        """Extract text from a single PIL Image using the best available backend."""
        preprocessed = self._preprocess_image(image)

        # Priority 1: EasyOCR (deep learning — better for complex layouts)
        if self._easyocr_reader:
            try:
                img_array = np.array(preprocessed)
                results = self._easyocr_reader.readtext(img_array)
                texts = [text for (_, text, conf) in results if conf > 0.1]
                extracted = " ".join(texts).strip()
                if extracted:
                    return extracted
            except Exception as exc:
                logger.warning("EasyOCR failed, trying Tesseract: %s", exc)

        # Priority 2: Tesseract
        if self._tesseract_available:
            try:
                import pytesseract  # type: ignore

                text = str(pytesseract.image_to_string(preprocessed, config="--psm 6"))
                return text.strip()
            except Exception as exc:
                logger.warning("Tesseract failed: %s", exc)

        raise OCRExtractionError(
            "No OCR backend available. Install Tesseract or EasyOCR."
        )

    # ── PDF Processing ───────────────────────────────────────────────

    def _extract_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text from PDF — tries native text extraction first (PyMuPDF),
        then falls back to OCR-based extraction (pdf2image + OCR engine).
        """
        # Strategy 1: PyMuPDF native text extraction (for digital PDFs)
        try:
            import fitz  # type: ignore

            doc = fitz.open(pdf_path)
            pages = []
            for page_num, page in enumerate(doc, start=1):
                text = page.get_text().strip()
                if text:
                    pages.append(text)
                    logger.debug("PDF page %d: extracted %d chars (native)", page_num, len(text))
            doc.close()

            if pages:
                return "\n\n--- Page Break ---\n\n".join(pages)
        except ImportError:
            logger.info("PyMuPDF not installed — using OCR-based PDF extraction")
        except Exception as exc:
            logger.warning("PyMuPDF failed: %s — falling back to OCR", exc)

        # Strategy 2: Convert PDF pages to images, then OCR each page
        try:
            from pdf2image import convert_from_path

            images = convert_from_path(pdf_path, poppler_path=self.poppler_path)  # type: ignore
            if not images:
                raise OCRExtractionError("PDF appears empty or unreadable")

            pages = []
            for page_num, image in enumerate(images, start=1):
                logger.debug("OCR processing PDF page %d...", page_num)
                pages.append(self._extract_from_image(image))

            return "\n\n--- Page Break ---\n\n".join(pages)
        except OCRExtractionError:
            raise
        except Exception as exc:
            raise OCRExtractionError(
                f"PDF processing failed. Ensure Poppler is installed. Error: {exc}"
            )
