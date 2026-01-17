"""
Preprocessing module for PDF handling and OCR.
Handles detection of digital vs scanned PDFs and text extraction.
"""
import re
import os
from typing import Tuple, Optional
import pdfplumber
from paddleocr import PaddleOCR

from .utils import logger

# Initialize PaddleOCR engine (lazy loading)
_ocr_engine: Optional[PaddleOCR] = None

def get_ocr_engine() -> PaddleOCR:
    """Get or initialize the OCR engine (singleton pattern)."""
    global _ocr_engine
    if _ocr_engine is None:
        logger.info("Initializing PaddleOCR engine...")
        _ocr_engine = PaddleOCR(use_angle_cls=True, lang='es', show_log=False)
    return _ocr_engine


def is_digital_pdf(pdf_path: str) -> bool:
    """
    Check if a PDF is digital (has selectable text) or scanned.
    
    Args:
        pdf_path: Path to the PDF file.
        
    Returns:
        True if the PDF has extractable text, False if it's scanned/image-based.
    """
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text and len(text.strip()) > 50:  # Meaningful text threshold
                    return True
        return False
    except Exception as e:
        logger.error("Error checking PDF type", error=str(e), path=pdf_path)
        return False


def extract_text_from_digital_pdf(pdf_path: str) -> Tuple[str, float]:
    """
    Extract text from a digital PDF using pdfplumber.
    
    Args:
        pdf_path: Path to the PDF file.
        
    Returns:
        Tuple of (extracted_text, confidence_score).
        Confidence is 1.0 for digital PDFs as text is directly extracted.
    """
    try:
        text_parts = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
        
        full_text = '\n'.join(text_parts)
        logger.info("Extracted text from digital PDF", 
                   path=pdf_path, 
                   characters=len(full_text))
        return full_text, 1.0
    except Exception as e:
        logger.error("Error extracting text from digital PDF", 
                    error=str(e), 
                    path=pdf_path)
        return "", 0.0


def extract_text_with_ocr(pdf_path: str) -> Tuple[str, float]:
    """
    Extract text from a scanned PDF using PaddleOCR.
    
    Args:
        pdf_path: Path to the PDF file.
        
    Returns:
        Tuple of (extracted_text, average_confidence_score).
    """
    try:
        ocr = get_ocr_engine()
        result = ocr.ocr(pdf_path, cls=True)
        
        if not result:
            logger.warning("OCR returned no results", path=pdf_path)
            return "", 0.0
        
        text_parts = []
        confidence_scores = []
        
        for page_result in result:
            if page_result is None:
                continue
            for line in page_result:
                if line and len(line) >= 2:
                    text_info = line[1]
                    if isinstance(text_info, tuple) and len(text_info) >= 2:
                        text_parts.append(text_info[0])
                        confidence_scores.append(text_info[1])
        
        full_text = ' '.join(text_parts)
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
        
        logger.info("Extracted text with OCR", 
                   path=pdf_path, 
                   characters=len(full_text),
                   avg_confidence=avg_confidence)
        
        return full_text, avg_confidence
    except Exception as e:
        logger.error("Error extracting text with OCR", 
                    error=str(e), 
                    path=pdf_path)
        return "", 0.0


def extract_text(pdf_path: str) -> Tuple[str, float, str]:
    """
    Main entry point: Extract text from a PDF, automatically detecting type.
    
    Args:
        pdf_path: Path to the PDF file.
        
    Returns:
        Tuple of (extracted_text, confidence_score, extraction_method).
        extraction_method is either 'digital' or 'ocr'.
    """
    if not os.path.exists(pdf_path):
        logger.error("PDF file not found", path=pdf_path)
        return "", 0.0, "error"
    
    if is_digital_pdf(pdf_path):
        text, confidence = extract_text_from_digital_pdf(pdf_path)
        return text, confidence, "digital"
    else:
        text, confidence = extract_text_with_ocr(pdf_path)
        return text, confidence, "ocr"


def clean_text(text: str) -> str:
    """
    Clean and normalize extracted text.
    
    Args:
        text: Raw extracted text.
        
    Returns:
        Cleaned and normalized text.
    """
    if not text:
        return ""
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove control characters but keep newlines
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    
    # Normalize common OCR errors
    ocr_corrections = {
        'l': 'I',  # Only in specific contexts, handled carefully
        '0': 'O',  # Only in specific contexts
    }
    # Note: These corrections are context-sensitive and applied in extraction.py
    
    return text.strip()


def correct_ocr_errors(text: str, context: str = "general") -> str:
    """
    Apply context-sensitive OCR error corrections.
    
    Args:
        text: Text to correct.
        context: Context hint ('numeric', 'alpha', 'rfc', 'general').
        
    Returns:
        Corrected text.
    """
    if context == "numeric":
        # In numeric context, replace letters that look like numbers
        corrections = {'O': '0', 'o': '0', 'I': '1', 'l': '1', 'S': '5', 'B': '8'}
        for wrong, right in corrections.items():
            text = text.replace(wrong, right)
    elif context == "alpha":
        # In alpha context, replace numbers that look like letters
        corrections = {'0': 'O', '1': 'I', '5': 'S', '8': 'B'}
        for wrong, right in corrections.items():
            text = text.replace(wrong, right)
    
    return text
