# Simulación de Flujo de Datos y Casos de Prueba

Este documento simula el comportamiento interno del sistema ante 6 escenarios distintos, analizando la toma de decisiones, la trazabilidad del dato y detectando áreas de mejora.

---

## Caso 1: Factura Digital Perfecta (Camino Feliz)
- **Entrada:** PDF generado por software contable (nativo).
- **Flujo:**
    1.  **Pre-procesamiento:** `is_digital_pdf` detecta caracteres. Usa `pdfplumber`.
    2.  **Clasificación:** El modelo ML identifica "factura" con confianza **0.99**.
        -   **¿Cómo se genera el 0.99?**: Al ser texto digital nativo, el `TfidfVectorizer` detecta bigramas exactos como `"Folio Fiscal"` o `"Sello Digital"`. El clasificador **SVM** (SVC) mapea este vector a una zona de máxima certeza. Gracias al **Escalado de Platt** (`probability=True`), la distancia al hiperplano se traduce en una probabilidad logística casi perfecta (0.99).
    3.  **Extracción:**
        -   RFC: Encontrado por **Regex**.
        -   Monto: Encontrado por **Regex**.
        -   Proveedor: Identificado por **spaCy NER**.
- **Trazabilidad:**
    ```json
    "traceability": { "rfc": "regex", "proveedor": "spacy_ner", "monto_total": "regex" }
    ```
- **Resultado:** Éxito total.

---

## Caso 2: Comprobante SPEI Escaneado (Ruta OCR)
- **Entrada:** Foto de un recibo impreso de transferencia bancaria.
- **Flujo:**
    1.  **Pre-procesamiento:** No hay texto seleccionable. Activa **PaddleOCR**.
    2.  **Limpieza:** `correct_ocr_errors` cambia "O" por "0" en montos.
    3.  **Clasificación:** Keywords encuentran "SPEI" y "referencia". Tipo: "comprobante".
    4.  **Extracción:** Regex no encuentra el RFC por ruido en el escaneo. Se llama a **Ollama**.
- **Trazabilidad:**
    ```json
    "traceability": { "rfc": "llm", "monto_total": "regex", "_extraction_method": "ocr" }
    ```
- **Resultado:** Datos recuperados gracias al fallback de IA.

---

## Caso 3: Factura con Formato Atípico (Ruta Híbrida)
- **Entrada:** Factura extranjera o con diseño personalizado.
- **Flujo:**
    1.  **Extracción:** Regex falla al identificar quién es el emisor y quién el receptor por la posición inusual de las etiquetas.
    2.  **Decisión:** El sistema detecta ambigüedad y envía el texto a **Ollama (Llama3)** con el prompt de contexto.
- **Trazabilidad:**
    ```json
    "traceability": { "emisor": "llm_inference", "receptor": "llm_inference" }
    ```
- **Resultado:** Extracción exitosa mediante inferencia semántica.

---

## Caso 4: CASO BORDE - Documento con Baja Resolución (Mejora de Diseño)
- **Entrada:** PDF escaneado a 72dpi, muy borroso.
- **Flujo:**
    1.  **OCR:** PaddleOCR genera texto con muchas "alucinaciones" (ej. "T0TAL" en vez de "TOTAL").
    2.  **Validación:** El monto extraído no es un número válido. `validate_fields` devuelve `valid: false`.
- **Hallazgo:** El sistema actual no intenta re-procesar con filtros de imagen.
- **Mejora sugerida:** Implementar una capa de pre-procesamiento de imagen (OpenCV) para mejorar el contraste antes del OCR si la confianza inicial es baja (< 0.6).

---

## Caso 5: CASO BORDE - Documento Multimodal (Desconocido)
- **Entrada:** Una carta de agradecimiento que incluye un RFC pero no es una factura.
- **Flujo:**
    1.  **Clasificación:** El modelo ML da una confianza baja (0.35) para todos los tipos.
    2.  **Extracción:** Extrae el RFC (porque la regex lo encuentra), pero al no tener tipo definido, la validación de negocio podría fallar.
- **Hallazgo:** El sistema extrae datos de cualquier cosa que parezca una factura.
- **Mejora sugerida:** Implementar un "Threshold de Confianza" en la clasificación. Si es < 0.5, rechazar el documento antes de gastar recursos en extracción de campos.

---

## Caso 6: CASO BORDE - Conflicto de RFCs (Ambigüedad)
- **Entrada:** Documento que menciona un RFC de un tercero (ej. "Enviado por logística S.A. de C.V. RFC: XXXX") además del Emisor y Receptor.
- **Flujo:**
    1.  **Regex:** Encuentra 3 RFCs distintos. ¿Cuál elige?
    2.  **Lógica actual:** Toma el primero que encuentra.
- **Hallazgo:** Riesgo de asignar el RFC de la empresa de logística como el emisor de la factura.
- **Mejora sugerida:** Mejorar la lógica de `extract_fields` para que, si hay > 2 RFCs, se fuerce siempre la consulta al LLM para que este determine el "rol" de cada RFC basándose en el lenguaje circundante.

---

## Resumen de Trazabilidad Generada por el Sistema

El sistema genera una "huella digital" de la extracción en el objeto de salida:

1.  **`_extraction_method`**: "digital" (rápido/exacto) o "ocr" (procesamiento pesado).
2.  **`_ocr_confidence`**: Valor del motor de visión (PaddleOCR).
3.  **`source` por campo**:
    -   `regex`: Alta confianza técnica.
    -   `spacy_ner`: Confianza estadística/lingüística.
    -   `llm`: Confianza semántica (inferencia consciente).

Este diseño permite que el usuario humano sepa cuándo debe revisar un dato con mayor atención (especialmente si la fuente es `llm` o `ocr` con baja confianza).
