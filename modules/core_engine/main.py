from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .crawler import stream_smart_ingest
from .pdf_ingest import save_and_parse_pdf

app = FastAPI(title="Cognitive Kitchen API", version="0.0.1")

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
    Receives an uploaded PDF file, extracts text using PyMuPDF, and stores it in the local vault.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    try:
        file_bytes = await file.read()
        result = save_and_parse_pdf(file_bytes, file.filename)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))