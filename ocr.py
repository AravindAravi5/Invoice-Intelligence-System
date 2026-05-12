import os
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
from pdf2image import convert_from_path
import logging

# Configure basic logging for error tracking
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class OCRExtractor:
    def __init__(self, tesseract_cmd: str = None, poppler_path: str = None):
        """
        Initialize the OCR Extractor.
        """
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        self.poppler_path = poppler_path
        self.easyocr_reader = None
        
        # Pre-initialize EasyOCR to speed up subsequent requests
        try:
            import easyocr
            self.easyocr_reader = easyocr.Reader(['en'], verbose=False)
            logging.info("EasyOCR initialized successfully (verbose=False)")
        except Exception as e:
            logging.warning(f"EasyOCR initialization failed: {e}")

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

    def _extract_from_image(self, image: Image.Image) -> str:
        """
        Extract text from a single PIL Image using easyocr or tesseract.
        """
        try:
            # Apply preprocessing steps to improve quality
            preprocessed_img = self._preprocess_image(image)
            
            # Try easyocr first if initialized
            if self.easyocr_reader:
                import numpy as np
                img_array = np.array(preprocessed_img)
                results = self.easyocr_reader.readtext(img_array)
                texts = [text for (bbox, text, confidence) in results if confidence > 0.1]
                return ' '.join(texts).strip()
            else:
                # Use pytesseract to extract text
                text = pytesseract.image_to_string(preprocessed_img, config='--psm 6')
                return text.strip()
        except Exception as e:
            # Safely handle potential non-encodable characters in exception messages
            err_msg = str(e).encode('ascii', 'ignore').decode('ascii')
            logging.error(f"OCR extraction failed: {err_msg}")
            raise RuntimeError(f"OCR processing failed on image: {err_msg}")

    def _extract_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text from a PDF file using PyMuPDF (fitz) or fallback to pdf2image.
        """
        try:
            import fitz
            doc = fitz.open(pdf_path)
            extracted_text = []
            for page_num, page in enumerate(doc, start=1):
                logging.info(f"Processing PDF page {page_num}...")
                text = page.get_text()
                extracted_text.append(text)
            return "\n\n--- Page Break ---\n\n".join(extracted_text)
        except Exception as e:
            logging.warning(f"PyMuPDF failed, falling back to pdf2image: {e}")
            try:
                images = convert_from_path(pdf_path, poppler_path=self.poppler_path)
                if not images:
                    raise ValueError("PDF appears to be empty or unreadable.")
                extracted_text = []
                for page_num, image in enumerate(images, start=1):
                    logging.info(f"Processing PDF page {page_num}...")
                    text = self._extract_from_image(image)
                    extracted_text.append(text)
                return "\n\n--- Page Break ---\n\n".join(extracted_text)
            except Exception as e2:
                logging.error(f"PDF processing failed: {e2}")
                raise RuntimeError(f"Failed to process PDF file. Ensure 'poppler' is installed. Error: {str(e2)}")

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
                # Handle unsupported formats gracefully
                raise ValueError(f"Unsupported file format: {file_ext}. Supported formats: {supported_images + ['.pdf']}")
                
        except Exception as e:
            # Catch any unexpected errors during extraction
            logging.error(f"Extraction process failed for {file_path}: {e}")
            raise

# Example Usage
if __name__ == "__main__":
    # Note for Windows users: You may need to provide paths to tesseract and poppler
    # tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    # poppler_path = r'C:\Program Files\poppler-xx\bin'
    # extractor = OCRExtractor(tesseract_cmd=tesseract_cmd, poppler_path=poppler_path)
    
    extractor = OCRExtractor()
    
    test_file = "sample_invoice.pdf" # Replace with a real path
    
    if os.path.exists(test_file):
        try:
            print(f"Extracting text from {test_file}...")
            result = extractor.extract_text(test_file)
            print("\n--- Extracted Text ---\n")
            print(result)
        except Exception as e:
            print(f"\nAn error occurred: {e}")
    else:
        print(f"Please place a '{test_file}' in the directory to test the script.")
