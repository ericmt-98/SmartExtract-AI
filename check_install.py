
import sys
print("Checking imports...")
try:
    import fastapi
    print("FastAPI ok")
    import paddleocr
    print("PaddleOCR ok")
    import spacy
    print("SpaCy ok")
    import app.main
    print("App module ok")
except Exception as e:
    print(f"IMPORT ERROR: {e}")
    sys.exit(1)

print("Imports successful.")
