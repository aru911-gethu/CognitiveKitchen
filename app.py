import sys
import os

# --- PATH INJECTOR FIX ---
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
# -------------------------

import streamlit as st
import requests
import json
import pandas as pd
from dotenv import load_dotenv
from modules.core_engine.part_b_rag.retriever import get_rag_chain

load_dotenv()

# --- CACHE THE RAG CHAIN (Loads ONCE into RAM for zero lag) ---
@st.cache_resource
def get_cached_rag_chain():
    return get_rag_chain()

def render_web_ingest():
    st.subheader("Smart Web Crawler (Live Streaming)")
    target_url = st.text_input("Recipe or Category Webpage URL", placeholder="https://chitrasfoodbook.com/...")

    if st.button("Process Web URL Live"):
        if not target_url:
            st.warning("Please enter a URL first!")
            return
            
        status_box = st.status("Initializing crawler...", expanded=True)
        progress_bar = st.progress(0)
        scraped_recipes = []
        table_placeholder = st.empty()
        
        try:
            response = requests.post(
                "http://127.0.0.1:8000/api/v1/stream-ingest",
                json={"url": target_url},
                stream=True
            )
            
            for line in response.iter_lines():
                if line:
                    data = json.loads(line.decode("utf-8"))
                    msg_status = data.get("status")
                    
                    if msg_status == "info":
                        status_box.write(data.get("message"))
                    elif msg_status == "progress":
                        current, total, recipe = data.get("current"), data.get("total"), data.get("recipe")
                        scraped_recipes.append(recipe)
                        
                        progress_bar.progress(current / total)
                        status_box.update(label=f"Scraping [{current}/{total}]: {recipe.get('title')}", state="running")
                        
                        df = pd.DataFrame(scraped_recipes)
                        table_placeholder.dataframe(df[["title", "saved_path"]], use_container_width=True)
                        
                    elif msg_status == "complete":
                        status_box.update(label="Vault Ingestion Complete! ✅", state="complete")
                        progress_bar.progress(1.0)
        except Exception as e:
            st.error(f"Connection error: {e}")

def render_pdf_ingest():
    st.subheader("Upload Recipe PDF")
    uploaded_file = st.file_uploader("Choose a recipe PDF file", type=["pdf"])

    if uploaded_file is not None and st.button("Upload & Parse PDF"):
        with st.spinner("Uploading PDF, extracting text with PyMuPDF, and saving to vault..."):
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                response = requests.post("http://127.0.0.1:8000/api/v1/ingest-pdf", files=files)
                
                if response.status_code == 200:
                    res_data = response.json()
                    st.success(f"Successfully ingested PDF: **{res_data.get('title')}**")
                    st.json(res_data)
                else:
                    st.error(f"Server error: {response.text}")
            except Exception as e:
                st.error(f"Failed to connect to backend: {e}")

def render_chat():
    st.subheader("Talk to Your Recipe Vault")
    st.caption("Ask questions about cooking times, ingredients, and substitutions directly from your ingested data.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_prompt := st.chat_input("E.g., How long should I pressure cook the dal?"):
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        with st.chat_message("assistant"):
            try:
                # Use the cached version for fast response times!
                rag_chain = get_cached_rag_chain()
                response_placeholder = st.empty()
                full_response = ""

                for chunk in rag_chain.stream(user_prompt):
                    full_response += chunk
                    response_placeholder.markdown(full_response + "▌")

                response_placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})

            except FileNotFoundError:
                st.warning("Vector index not found. Please run `vector_builder.py` first.")
            except Exception as e:
                st.error(f"Inference error: {str(e)}")

def main():
    st.set_page_config(page_title="Cognitive Kitchen", page_icon="🍳", layout="centered")
    st.title("🍳 Cognitive Kitchen")
    st.markdown("Your AI-powered multi-modal recipe vault.")

    tab1, tab2, tab3 = st.tabs(["🌐 Web Ingest", "📄 PDF Ingest", "🧑‍🍳 Recipe Assistant Chat"])

    with tab1:
        render_web_ingest()
    with tab2:
        render_pdf_ingest()
    with tab3:
        render_chat()

if __name__ == "__main__":
    main()