"""
Main FastAPI application for Intelligent Data Extraction System.
Provides REST API endpoints for processing PDF documents.
"""
import os
import uuid
import tempfile
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse

from .models import ExtractedData, ExtractionResponse
from .preprocessing import extract_text, clean_text
from .extraction import classify_document, extract_fields, validate_fields
from .utils import configure_logging, logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown."""
    configure_logging()
    logger.info("Application starting up...")
    yield
    logger.info("Application shutting down...")


app = FastAPI(
    title="Sistema de Extracción Inteligente de Datos",
    description="""
    API para extraer datos estructurados de facturas, comprobantes y recibos en PDF.
    
    ## Características
    - Detección automática de PDF digital vs escaneado
    - OCR con PaddleOCR para documentos escaneados
    - Extracción híbrida: Regex → spaCy NER → LLM (Ollama)
    - Validación de campos extraídos
    - Trazabilidad completa de cada campo
    """,
    version="1.0.0",
    lifespan=lifespan
)


def cleanup_file(path: str):
    """Background task to clean up temporary files."""
    try:
        if os.path.exists(path):
            os.remove(path)
            logger.info("Cleaned up temporary file", path=path)
    except Exception as e:
        logger.warning("Failed to cleanup file", path=path, error=str(e))


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": "Intelligent Data Extraction API",
        "version": "1.0.0"
    }


@app.get("/health")
async def health_check():
    """Detailed health check."""
    return {
        "status": "healthy",
        "components": {
            "api": "ok",
            "ocr": "ok",  # Could add actual health checks
            "llm": "check_connection"  # Could ping Ollama
        }
    }


@app.post("/extract", response_model=ExtractionResponse)
async def extract_from_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="PDF file to process")
):
    """
    Extract structured data from a PDF document.
    
    ## Process Flow
    1. Receive PDF upload
    2. Detect if digital or scanned
    3. Extract text (OCR if needed)
    4. Classify document type
    5. Extract fields (RFC, dates, amounts, etc.)
    6. Validate extracted data
    7. Return structured JSON response
    
    ## Response
    Returns extracted fields with traceability information showing
    the source of each field (regex, spacy_ner, llm).
    """
    # Validate file type
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=400, 
            detail="Only PDF files are supported"
        )
    
    # Generate unique document ID for traceability
    document_id = str(uuid.uuid4())
    logger.info("Processing document", document_id=document_id, filename=file.filename)
    
    # Save uploaded file temporarily
    temp_path = None
    try:
        # Create temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name
        
        logger.info("File saved temporarily", path=temp_path, size=len(content))
        
        # Step 1: Extract text
        raw_text, ocr_confidence, extraction_method = extract_text(temp_path)
        
        if not raw_text or len(raw_text.strip()) < 10:
            raise HTTPException(
                status_code=422,
                detail="Could not extract text from PDF. The file may be empty or corrupted."
            )
        
        # Step 2: Clean text
        cleaned_text = clean_text(raw_text)
        
        # Step 3: Classify document
        doc_type, classification_confidence = classify_document(cleaned_text)
        
        # Step 4: Extract fields
        fields, traceability = extract_fields(cleaned_text, doc_type)
        
        # Add extraction method to traceability
        traceability["_extraction_method"] = extraction_method
        traceability["_ocr_confidence"] = str(ocr_confidence) if extraction_method == "ocr" else "N/A"
        traceability["_document_id"] = document_id
        
        # Step 5: Validate
        is_valid, issues = validate_fields(fields)
        
        # Build response
        extracted_data = ExtractedData(
            tipo_documento=fields.get("tipo_documento", "desconocido"),
            proveedor=fields.get("proveedor"),
            rfc=fields.get("rfc"),
            fecha=fields.get("fecha"),
            monto_total=fields.get("monto_total"),
            moneda=fields.get("moneda", "MXN"),
            numero_referencia=fields.get("numero_referencia"),
            banco_emisor=fields.get("banco_emisor"),
            traceability=traceability,
            confidence_scores={}
        )
        
        response = ExtractionResponse(
            data=extracted_data,
            valid=is_valid,
            issues=issues
        )
        
        logger.info(
            "Extraction complete",
            document_id=document_id,
            doc_type=doc_type,
            valid=is_valid,
            fields_extracted=len([f for f in fields.values() if f is not None])
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Extraction failed", document_id=document_id, error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Error processing document: {str(e)}"
        )
    finally:
        # Schedule cleanup of temporary file
        if temp_path:
            background_tasks.add_task(cleanup_file, temp_path)


@app.post("/extract/batch")
async def extract_batch(files: list[UploadFile] = File(...)):
    """
    Process multiple PDF documents in batch.
    
    Note: For large batches, consider using async processing with a queue.
    """
    if len(files) > 10:
        raise HTTPException(
            status_code=400,
            detail="Maximum 10 files per batch. For larger batches, use individual requests."
        )
    
    results = []
    for file in files:
        try:
            # Reuse single extraction logic
            background_tasks = BackgroundTasks()
            result = await extract_from_pdf(background_tasks, file)
            results.append({
                "filename": file.filename,
                "success": True,
                "data": result
            })
        except HTTPException as e:
            results.append({
                "filename": file.filename,
                "success": False,
                "error": e.detail
            })
    
    return {"results": results, "total": len(files), "successful": sum(1 for r in results if r["success"])}


# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors."""
    logger.error("Unhandled exception", error=str(exc), path=str(request.url))
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal error occurred. Please try again."}
    )
