from pydantic import BaseModel
from typing import Optional, Dict

class TaxActor(BaseModel):
    nombre: Optional[str] = None
    rfc: Optional[str] = None

class ExtractedData(BaseModel):
    tipo_documento: str
    emisor: Optional[TaxActor] = None
    receptor: Optional[TaxActor] = None
    fecha: Optional[str] = None
    monto_total: Optional[float] = None
    moneda: Optional[str] = "MXN"
    numero_referencia: Optional[str] = None
    banco_emisor: Optional[str] = None
    traceability: Dict[str, str] = {}
    confidence_scores: Dict[str, float] = {}

class ExtractionResponse(BaseModel):
    data: ExtractedData
    valid: bool
    issues: list[str] = []
