import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../../"))
VECTOR_STORE_DIR = os.path.join(ROOT_DIR, "data", "vector_store")

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def format_retrieved_docs(docs):
    formatted_chunks = []
    for doc in docs:
        source = doc.metadata.get("title") or doc.metadata.get("source") or "Recipe Vault"
        formatted_chunks.append(f"--- Source: {source} ---\n{doc.page_content}")
    return "\n\n".join(formatted_chunks)

def get_rag_chain():
    if not os.path.exists(os.path.join(VECTOR_STORE_DIR, "index.faiss")):
        raise FileNotFoundError("FAISS index not found. Run vector_builder.py first.")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"}
    )
    
    vectorstore = FAISS.load_local(
        VECTOR_STORE_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )
    
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are 'Cognitive Kitchen AI', an expert culinary assistant.\n"
         "You have access to a recipe vault context below. Use the recipes found in the context to answer the user's request, suggest dishes, or explain ingredients and steps.\n"
         "If the context contains relevant recipes, summarize or present them clearly. If the information is completely unrelated to the request, politely state what is available in the vault.\n\n"
         "Context:\n{context}"),
        ("human", "{question}")
    ])
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.2,
        streaming=True
    )

    chain = (
        {"context": retriever | format_retrieved_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain