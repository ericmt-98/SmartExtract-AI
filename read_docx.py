
import zipfile
import xml.etree.ElementTree as ET
import sys
import os

def read_docx(file_path):
    try:
        if not os.path.exists(file_path):
            print(f"Error: File not found: {file_path}")
            return

        with zipfile.ZipFile(file_path) as z:
            xml_content = z.read('word/document.xml')
        
        tree = ET.fromstring(xml_content)
        namespace = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
        
        text = []
        for p in tree.findall('.//w:p', namespace):
            paragraph_text = []
            for t in p.findall('.//w:t', namespace):
                if t.text:
                    paragraph_text.append(t.text)
            if paragraph_text:
                text.append(''.join(paragraph_text))
        
        return '\n'.join(text)
    except Exception as e:
        return f"Error reading {file_path}: {e}"

if __name__ == "__main__":
    files = [
        "PROBLEMA REAL EXTRACCIÓN INTELIGENTE DE DATOS DE FACTURAS Y COMPROBANTES DE PAGO.docx",
        "Arquitectura.docx",
        "Guia .docx"
    ]
    
    with open("extracted_text.txt", "w", encoding="utf-8") as f:
        for filename in files:
            f.write(f"--- START OF {filename} ---\n")
            f.write(read_docx(filename))
            f.write(f"\n--- END OF {filename} ---\n\n")
    print("Extraction complete.")
