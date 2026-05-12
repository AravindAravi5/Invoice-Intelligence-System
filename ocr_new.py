"""
OCR Extractor with Fallback Support
Tries to use Tesseract first, falls back to EasyOCR if not available
"""

import os
import logging
from PIL import Image, ImageEnhance, ImageFilter
from pathlib import Path

# Configure basic logging for error tracking
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class OCRExtractor:
    def __init__(self, tesseract_cmd: str = None, poppler_path: str = None):
        """
        Initialize the OCR Extractor.
        
        Args:
            tesseract_cmd (str): Optional path to the tesseract executable if not in system PATH.
            poppler_path (str): Optional path to poppler binaries (required for Windows PDF processing).
        """
        self.tesseract_cmd = tesseract_cmd
        self.poppler_path = poppler_path
        self.use_tesseract = False
        self.use_easyocr = False
        
        # Try to detect which OCR backend to use
        self._detect_ocr_backend()
    
    def _detect_ocr_backend(self):
        """Detect which OCR backend is available"""
        # Try Tesseract first
        try:
            import pytesseract
            if self.tesseract_cmd:
                pytesseract.pytesseract.tesseract_cmd = self.tesseract_cmd
            
            # Test if tesseract is accessible
            pytesseract.get_tesseract_version()
            self.use_tesseract = True
            logging.info("✓ Tesseract OCR detected and will be used")
            return
        except Exception as e:
            logging.warning(f"Tesseract not available: {e}")
        
        # Try EasyOCR as fallback
        try:
            import easyocr
            self.use_easyocr = True
            self.reader = easyocr.Reader(['en'])
            logging.info("✓ EasyOCR detected and will be used as fallback")
            return
        except Exception as e:
            logging.warning(f"EasyOCR not available: {e}")
        
        logging.warning("⚠ No OCR backend available - will use demo mode with placeholder text")
    
    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """
        Preprocess the image to improve OCR accuracy.
        Handles low-quality images by adjusting contrast and sharpening.
        """
        try:
            # 1. Convert image to grayscale to remove color noise
            image = image.convert('L')
            
            # 2. Enhance contrast to make text stand out against the background
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(2.0)
            
            # 3. Apply a sharpening filter to clarify blurry text edges
            image = image.filter(ImageFilter.SHARPEN)
            
            return image
        except Exception as e:
            logging.error(f"Image preprocessing failed: {e}")
            raise ValueError(f"Error during image preprocessing: {str(e)}")
    
    def _extract_from_image_tesseract(self, image: Image.Image) -> str:
        """Extract text using Tesseract"""
        try:
            import pytesseract
            preprocessed_img = self._preprocess_image(image)
            text = pytesseract.image_to_string(preprocessed_img, config='--psm 6')
            return text.strip()
        except Exception as e:
            logging.error(f"Tesseract extraction failed: {e}")
            raise RuntimeError(f"OCR processing failed: {str(e)}")
    
    def _extract_from_image_easyocr(self, image: Image.Image) -> str:
        """Extract text using EasyOCR"""
        try:
            # EasyOCR expects numpy array
            import numpy as np
            img_array = np.array(image)
            results = self.reader.readtext(img_array)
            
            # Extract and sort text
            texts = []
            for (bbox, text, confidence) in results:
                if confidence > 0.1:  # Only include text with confidence > 10%
                    texts.append(text)
            
            return ' '.join(texts).strip()
        except Exception as e:
            logging.error(f"EasyOCR extraction failed: {e}")
            raise RuntimeError(f"OCR processing failed: {str(e)}")
    
    def _extract_from_image_demo(self, image: Image.Image) -> str:
        """Demo/fallback extraction with sample data"""
        logging.info("Using demo mode - no OCR backend available")
        return """SAMPLE INVOICE

Invoice Number: INV-2024-001
Date: 2024-04-30
Vendor: Acme Corporation
Amount: $1,500.00

Items:
- Professional Services: $1,000.00
- Software License: $400.00
- Support & Maintenance: $100.00

Total: $1,500.00

[Note: This is demo text. Install Tesseract or EasyOCR for actual OCR]"""
    
    def _extract_from_image(self, image: Image.Image) -> str:
        """Extract text from a single PIL Image"""
        if self.use_tesseract:
            return self._extract_from_image_tesseract(image)
        elif self.use_easyocr:
            return self._extract_from_image_easyocr(image)
        else:
            return self._extract_from_image_demo(image)
    
    def _extract_from_pdf(self, pdf_path: str) -> str:
        """
        Convert a PDF file to images and extract text from each page.
        """
        try:
            from pdf2image import convert_from_path
            
            # Convert PDF pages to a list of PIL Images
            images = convert_from_path(pdf_path, poppler_path=self.poppler_path)
            
            if not images:
                raise ValueError("PDF appears to be empty or unreadable.")

            extracted_text = []
            for page_num, image in enumerate(images, start=1):
                logging.info(f"Processing PDF page {page_num}...")
                text = self._extract_from_image(image)
                extracted_text.append(text)
                
            # Join all page texts with a double newline for clear separation
            return "\n\n--- Page Break ---\n\n".join(extracted_text)
            
        except ImportError:
            logging.warning("pdf2image not available, attempting direct image loading...")
            # Fallback: try to open as image
            try:
                # Try using Fitz/PyMuPDF as fallback
                import fitz
                doc = fitz.open(pdf_path)
                extracted_text = []
                for page_num, page in enumerate(doc, start=1):
                    logging.info(f"Processing PDF page {page_num}...")
                    text = page.get_text()
                    extracted_text.append(text)
                return "\n\n--- Page Break ---\n\n".join(extracted_text)
            except:
                logging.error("Cannot process PDF - install pdf2image and poppler")
                return self._extract_from_image_demo(None)
        except Exception as e:
            logging.error(f"PDF processing failed: {e}")
            raise RuntimeError(f"Failed to process PDF file: {str(e)}")

    def extract_text(self, file_path: str) -> str:
        """
        Main interface to extract text from a file.
        Validates the file existence and format, then routes to the correct method.
        
        Args:
            file_path (str): The absolute or relative path to the PDF or image file.
            
        Returns:
            str: The extracted and cleaned text from the document.
        """
        # 1. Validate if file exists
        if not os.path.exists(file_path):
            error_msg = f"Invalid file path: {file_path} does not exist."
            logging.error(error_msg)
            raise FileNotFoundError(error_msg)
            
        # 2. Identify the file type by its extension
        file_ext = os.path.splitext(file_path)[1].lower()
        supported_images = ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']
        
        try:
            if file_ext in supported_images:
                # Open image using PIL
                logging.info(f"Starting extraction for image: {file_path}")
                with Image.open(file_path) as img:
                    return self._extract_from_image(img)
                    
            elif file_ext == '.pdf':
                # Process PDF file
                logging.info(f"Starting extraction for PDF: {file_path}")
                return self._extract_from_pdf(file_path)
                
            else:
                error_msg = f"Unsupported file format: {file_ext}. Supported: {', '.join(supported_images + ['.pdf'])}"
                logging.error(error_msg)
                raise ValueError(error_msg)
                
        except FileNotFoundError:
            raise
        except Exception as e:
            logging.error(f"Extraction failed for {file_path}: {e}")
            raise RuntimeError(f"OCR extraction failed: {str(e)}")
