import os
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# Corrected absolute imports routing perfectly to Part A
from modules.core_engine.part_a_ingestion.crawler import stream_smart_ingest
from modules.core_engine.part_a_ingestion.pdf_ingest import extract_text_from_pdf, save_extracted_pdf_to_vault

app = FastAPI(title="Cognitive Kitchen API", version="0.0.1")

# Request schema for the web crawler endpoint
class URLRequest(BaseModel):
    url: str

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "kitchen_engine"}

@app.post("/api/v1/stream-ingest")
def stream_ingest(request: URLRequest):
    """
    Streams live progress updates while crawling and batch-scraping a web index or single recipe.
    """
    return StreamingResponse(stream_smart_ingest(request.url), media_type="application/x-ndjson")

@app.post("/api/v1/ingest-pdf")
async def ingest_pdf(file: UploadFile = File(...)):
    """
    Receives an uploaded PDF file, temporarily saves it, extracts text using PyMuPDF, 
    and stores it in the local structured JSON vault.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    try:
        # 1. Save uploaded bytes to a temporary file so PyMuPDF can open it
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
        
        # 2. Extract text using our Part A pipeline
        extracted_pages = extract_text_from_pdf(tmp_path)
        
        # 3. Save the extracted content into the JSON vault
        output_path = save_extracted_pdf_to_vault(extracted_pages, file.filename)
        
        # 4. Clean up the temporary file
        os.remove(tmp_path)
        
        return {
            "status": "success", 
            "title": file.filename, 
            "saved_path": output_path, 
            "total_pages": len(extracted_pages)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))