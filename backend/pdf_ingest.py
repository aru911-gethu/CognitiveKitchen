import os
from pathlib import Path
import json
import fitz  # PyMuPDF

# Ensure local raw PDF vault directory exists
RAW_PDF_DIR = Path("data/raw_pdfs")
RAW_PDF_DIR.mkdir(parents=True, exist_ok=True)

def save_and_parse_pdf(file_bytes, filename: str):
    """
    Saves an uploaded PDF locally to data/raw_pdfs/, extracts its text using PyMuPDF,
    and structures it like a recipe vault record.
    """
    file_path = RAW_PDF_DIR / filename
    
    # 1. Save the raw PDF file locally
    with open(file_path, "wb") as f:
        f.write(file_bytes)
        
    print(f"Saved uploaded PDF to: {file_path}")
    
    # 2. Extract text from the PDF using PyMuPDF (fitz)
    extracted_text = ""
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        for page in doc:
            extracted_text += page.get_text() + "\n"
    except Exception as e:
        print(f"Error reading PDF text with PyMuPDF: {e}")
        extracted_text = "Could not extract text from PDF."
        
    # 3. Create a structured JSON record for the vault
    recipe_record = {
        "title": filename.replace(".pdf", "").replace("_", " ").title(),
        "source_type": "pdf_upload",
        "filename": filename,
        "saved_path": str(file_path),
        "raw_text": extracted_text.strip()
    }
    
    # Save a metadata JSON copy in data/raw_recipes/ for uniformity
    json_output_path = Path("data/raw_recipes") / f"{file_path.stem}.json"
    json_output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_output_path, "w", encoding="utf-8") as jf:
        json.dump(recipe_record, jf, indent=4, ensure_ascii=False)
        
    return {
        "status": "success",
        "type": "pdf",
        "title": recipe_record["title"],
        "saved_path": str(json_output_path),
        "pdf_path": str(file_path)
    }