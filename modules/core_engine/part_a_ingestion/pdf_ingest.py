import os
import json
import fitz  # PyMuPDF
from typing import List, Dict, Any

# Note the path change here: we now go up THREE levels (../../../) because we are in part_a_ingestion
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
RAW_PDFS_DIR = os.path.join(ROOT_DIR, "data", "raw_pdfs")
RAW_RECIPES_DIR = os.path.join(ROOT_DIR, "data", "raw_recipes")

def extract_text_from_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    """Extracts text page-by-page from a PDF using PyMuPDF (fitz)."""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at: {pdf_path}")

    extracted_pages = []
    doc = fitz.open(pdf_path)
    filename = os.path.basename(pdf_path)

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text").strip()
        if text:
            extracted_pages.append({
                "source_file": filename,
                "page_number": page_num + 1,
                "content": text
            })
    doc.close()
    return extracted_pages

def save_extracted_pdf_to_vault(extracted_pages: List[Dict[str, Any]], base_filename: str) -> str:
    """Saves extracted PDF content as a normalized JSON inside data/raw_recipes/."""
    os.makedirs(RAW_RECIPES_DIR, exist_ok=True)
    output_filename = f"pdf_{os.path.splitext(base_filename)[0]}.json"
    output_path = os.path.join(RAW_RECIPES_DIR, output_filename)
    
    payload = {
        "source": base_filename,
        "type": "pdf_document",
        "total_pages": len(extracted_pages),
        "pages": extracted_pages
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        
    return output_path