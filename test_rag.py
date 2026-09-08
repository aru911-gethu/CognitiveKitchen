import os
import glob
import json
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

RAW_DIR = "data/raw_recipes"
json_files = glob.glob(os.path.join(RAW_DIR, "*.json"))
print(f"-> Found {len(json_files)} raw JSON files in vault.")

docs = []
for file_path in json_files:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if "url" in data and "raw_text" in data:
                title = data.get("title", "Untitled")
                raw_text = data.get("raw_text", "").strip()
                if "dosa" in raw_text.lower() or "dosa" in title.lower():
                    print(f"   [MATCH FOUND]: {title}")
                docs.append(Document(page_content=raw_text, metadata={"title": title}))
    except Exception as e:
        print(f"Error reading {file_path}: {e}")

print(f"-> Total loaded documents: {len(docs)}")

VECTOR_DIR = "data/vector_store"
if os.path.exists(os.path.join(VECTOR_DIR, "index.faiss")):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    vectorstore = FAISS.load_local(
        VECTOR_DIR, embeddings, allow_dangerous_deserialization=True
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    query = "suggest me dosa recipes"
    retrieved_docs = retriever.invoke(query)
    print(f"\n-> Query: \"{query}\"")
    print(f"-> Retrieved {len(retrieved_docs)} chunks from FAISS index:")
    for i, d in enumerate(retrieved_docs, 1):
        print(f"   [{i}] Source Title: {d.metadata.get('title')} | Snippet: {d.page_content[:150]}...")
else:
    print("-> FAISS index directory not found. Please run vector_builder.py.")