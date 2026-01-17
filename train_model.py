"""
Training script for Document Classification Model.
Builds a TF-IDF + SVM classifier from PDFs in training_data/ folders.
"""
import os
import sys
import joblib
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

# Add app to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from app.preprocessing import extract_text, clean_text

# Configuration
TRAINING_DATA_DIR = Path(__file__).parent / "training_data"
MODEL_OUTPUT_PATH = Path(__file__).parent / "model.pkl"
DOCUMENT_TYPES = ["facturas", "comprobantes", "recibos"]

def load_training_data():
    """
    Load and extract text from all PDFs in training_data/ subfolders.
    
    Returns:
        Tuple of (texts, labels)
    """
    texts = []
    labels = []
    
    for doc_type in DOCUMENT_TYPES:
        folder = TRAINING_DATA_DIR / doc_type
        if not folder.exists():
            print(f"Warning: Folder {folder} does not exist. Skipping.")
            continue
        
        pdf_files = list(folder.glob("*.pdf"))
        print(f"Found {len(pdf_files)} PDFs in {doc_type}/")
        
        for pdf_path in pdf_files:
            try:
                # Extract text from PDF
                raw_text, confidence, method = extract_text(str(pdf_path))
                cleaned = clean_text(raw_text)
                
                if len(cleaned) < 50:
                    print(f"  Skipping {pdf_path.name}: Too little text extracted.")
                    continue
                
                texts.append(cleaned)
                # Remove trailing 's' for singular label (facturas -> factura)
                label = doc_type.rstrip('s') if doc_type.endswith('s') else doc_type
                labels.append(label)
                print(f"  Loaded: {pdf_path.name} ({method}, {len(cleaned)} chars)")
                
            except Exception as e:
                print(f"  Error processing {pdf_path.name}: {e}")
    
    return texts, labels


def train_classifier(texts, labels):
    """
    Train a TF-IDF + SVM classification pipeline.
    
    Args:
        texts: List of document texts.
        labels: List of document type labels.
        
    Returns:
        Trained Pipeline object.
    """
    print(f"\nTraining classifier with {len(texts)} documents...")
    
    # Create pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words=None,  # Keep Spanish stopwords for now
            min_df=1,
            max_df=0.95
        )),
        ('classifier', SVC(
            kernel='linear',
            probability=True,
            C=1.0,
            class_weight='balanced'
        ))
    ])
    
    # Train
    pipeline.fit(texts, labels)
    
    # Cross-validation score (if enough samples)
    if len(texts) >= 5:
        scores = cross_val_score(pipeline, texts, labels, cv=min(5, len(texts)))
        print(f"Cross-validation accuracy: {scores.mean():.2%} (+/- {scores.std() * 2:.2%})")
    
    return pipeline


def main():
    print("=" * 50)
    print("Document Classification Model Training")
    print("=" * 50)
    
    # Check training data exists
    if not TRAINING_DATA_DIR.exists():
        print(f"Error: Training data directory not found: {TRAINING_DATA_DIR}")
        print("Please create the directory and add PDF files.")
        sys.exit(1)
    
    # Load data
    texts, labels = load_training_data()
    
    if len(texts) == 0:
        print("\nError: No valid training documents found!")
        print("Please add PDF files to the training_data/ subfolders:")
        for doc_type in DOCUMENT_TYPES:
            print(f"  - training_data/{doc_type}/")
        sys.exit(1)
    
    # Check minimum samples
    unique_labels = set(labels)
    print(f"\nDocument types found: {unique_labels}")
    
    if len(unique_labels) < 2:
        print("Warning: Only one document type found. Need at least 2 types for training.")
        print("The model will be trained but may not be useful.")
    
    # Train model
    model = train_classifier(texts, labels)
    
    # Save model
    joblib.dump(model, MODEL_OUTPUT_PATH)
    print(f"\nModel saved to: {MODEL_OUTPUT_PATH}")
    print("Restart the API server to load the new model.")
    
    # Test prediction
    print("\n--- Quick Test ---")
    test_text = texts[0][:500] if texts else ""
    if test_text:
        prediction = model.predict([test_text])[0]
        probabilities = model.predict_proba([test_text])[0]
        classes = model.classes_
        print(f"Sample prediction: {prediction}")
        print(f"Probabilities: {dict(zip(classes, probabilities))}")


if __name__ == "__main__":
    main()
