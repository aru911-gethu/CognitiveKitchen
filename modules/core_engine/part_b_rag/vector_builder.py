import os
import json
import glob
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../../"))
RAW_RECIPES_DIR = os.path.join(ROOT_DIR, "data", "raw_recipes")
VECTOR_STORE_DIR = os.path.join(ROOT_DIR, "data", "vector_store")

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def load_documents_from_vault():
    json_files = glob.glob(os.path.join(RAW_RECIPES_DIR, "*.json"))
    documents = []

    for file_path in json_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if data.get("type") == "pdf_document":
                source = data.get("source", os.path.basename(file_path))
                for page in data.get("pages", []):
                    content = page.get("content", "").strip()
                    if content:
                        documents.append(Document(
                            page_content=content,
                            metadata={"source": source, "type": "pdf", "page": page.get("page_number", 1)}
                        ))

            elif "url" in data and "raw_text" in data:
                title = data.get("title", "Untitled Recipe")
                url = data.get("url", "")
                raw_text = data.get("raw_text", "").strip()

                if raw_text:
                    content = f"Recipe Title: {title}\nSource: {url}\n\n{raw_text}"
                    documents.append(Document(
                        page_content=content,
                        metadata={"source": url, "title": title, "type": "web"}
                    ))

        except (json.JSONDecodeError, OSError) as e:
            print(f"Skipping corrupted file {file_path}: {e}")

    return documents  # Correctly placed outside the loop

def build_and_save_vectorstore():
    docs = load_documents_from_vault()

    if not docs:
        raise ValueError(f"No documents found in {RAW_RECIPES_DIR}. Run Part A ingestion first.")
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"}
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
    vectorstore.save_local(VECTOR_STORE_DIR)
    print(f"FAISS index successfully saved to: {VECTOR_STORE_DIR}")

if __name__ == "__main__":
    build_and_save_vectorstore()