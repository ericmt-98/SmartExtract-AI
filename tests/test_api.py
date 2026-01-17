"""
Tests for the Intelligent Data Extraction API.
"""
import pytest
from fastapi.testclient import TestClient
import os
import tempfile

# Import the app
from app.main import app
from app.extraction import (
    extract_rfc, 
    extract_date, 
    extract_amount, 
    classify_document,
    validate_rfc,
    validate_date
)
from app.preprocessing import clean_text, correct_ocr_errors

client = TestClient(app)


class TestHealthEndpoints:
    """Test health check endpoints."""
    
    def test_root_endpoint(self):
        response = client.get("/")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
    
    def test_health_endpoint(self):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"


class TestRFCExtraction:
    """Test RFC extraction and validation."""
    
    def test_extract_valid_rfc_persona_moral(self):
        text = "La empresa con RFC ABC123456XY9 emite esta factura"
        rfc, source = extract_rfc(text)
        assert rfc == "ABC123456XY9"
        assert source == "regex"
    
    def test_extract_valid_rfc_persona_fisica(self):
        text = "RFC del cliente: XAXX010101000"
        rfc, source = extract_rfc(text)
        assert rfc == "XAXX010101000"
        assert source == "regex"
    
    def test_extract_rfc_not_found(self):
        text = "Este texto no contiene ningún RFC válido"
        rfc, source = extract_rfc(text)
        assert rfc is None
        assert source == ""
    
    def test_validate_rfc_valid(self):
        assert validate_rfc("ABC123456XY9") == True
        assert validate_rfc("XAXX010101000") == True
    
    def test_validate_rfc_invalid(self):
        assert validate_rfc("INVALID") == False
        assert validate_rfc(None) == False
        assert validate_rfc("") == False


class TestDateExtraction:
    """Test date extraction."""
    
    def test_extract_date_dd_mm_yyyy(self):
        text = "Fecha: 15/01/2024"
        date, source = extract_date(text)
        assert date == "2024-01-15"
        assert source == "regex"
    
    def test_extract_date_iso_with_time(self):
        text = "Fecha de emisión: 2022-11-02T15:14:06"
        date, source = extract_date(text)
        assert date == "2022-11-02"
        assert source == "regex"

    def test_extract_date_yyyy_mm_dd(self):
        text = "Fecha de emisión: 2024-03-20"
        date, source = extract_date(text)
        assert date == "2024-03-20"
        assert source == "regex"
    
    def test_extract_date_spanish_format(self):
        text = "Emitido el 25 de marzo de 2024"
        date, source = extract_date(text)
        assert date == "2024-03-25"
        assert source == "regex"
    
    def test_validate_date_valid(self):
        assert validate_date("2024-01-15") == True
        assert validate_date("2023-12-31") == True
    
    def test_validate_date_invalid(self):
        assert validate_date("2024-13-01") == False  # Invalid month
        assert validate_date("invalid") == False
        assert validate_date(None) == False



class TestAmountExtraction:
    """Test amount extraction."""
    
    def test_extract_total_with_dollar_sign(self):
        text = "Total: $1,234.56"
        amount, source = extract_amount(text)
        assert amount == 1234.56
        assert source == "regex"
    
    def test_extract_monto(self):
        text = "Monto a pagar: 5000.00 MXN"
        amount, source = extract_amount(text)
        assert amount == 5000.0
        assert source == "regex"
    
    def test_extract_importe(self):
        text = "Importe total: $999.99"
        amount, source = extract_amount(text)
        assert amount == 999.99


class TestDocumentClassification:
    """Test document classification."""
    
    def test_classify_factura(self):
        text = "FACTURA CFDI Folio Fiscal UUID del emisor al receptor"
        doc_type, confidence = classify_document(text)
        assert doc_type == "factura"
        assert confidence > 0
    
    def test_classify_comprobante(self):
        text = "COMPROBANTE de transferencia bancaria SPEI CLABE destino"
        doc_type, confidence = classify_document(text)
        assert doc_type == "comprobante"
        assert confidence > 0
    
    def test_classify_recibo(self):
        text = "RECIBO de pago abono a cuenta saldo pendiente"
        doc_type, confidence = classify_document(text)
        assert doc_type == "recibo"


class TestTextCleaning:
    """Test text cleaning utilities."""
    
    def test_clean_text_whitespace(self):
        text = "Text   with    multiple     spaces"
        cleaned = clean_text(text)
        assert cleaned == "Text with multiple spaces"
    
    def test_correct_ocr_numeric_context(self):
        text = "1O5B"  # Should become 1058
        corrected = correct_ocr_errors(text, context="numeric")
        assert corrected == "1058"
    
    def test_correct_ocr_alpha_context(self):
        text = "H0USE"  # Should become HOUSE
        corrected = correct_ocr_errors(text, context="alpha")
        assert corrected == "HOUSE"


class TestAPIExtraction:
    """Test the main extraction endpoint."""
    
    def test_extract_requires_pdf(self):
        # Test with non-PDF file
        response = client.post(
            "/extract",
            files={"file": ("test.txt", b"Some text content", "text/plain")}
        )
        assert response.status_code == 400
        assert "PDF" in response.json()["detail"]
    
    def test_extract_empty_file_check(self):
        # Create a minimal empty PDF-like file (won't work as real PDF)
        response = client.post(
            "/extract",
            files={"file": ("test.pdf", b"%PDF-1.4", "application/pdf")}
        )
        # Should fail gracefully
        assert response.status_code in [422, 500]


# Run with: pytest tests/test_api.py -v
