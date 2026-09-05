import streamlit as st
import requests
import json
import pandas as pd

st.set_page_config(page_title="Cognitive Kitchen", page_icon="🍳", layout="centered")

st.title("🍳 Cognitive Kitchen")
st.markdown("Your AI-powered multi-modal recipe vault. Ingest recipes via **Smart Web Crawler** or **PDF Document Upload**.")

# Create clean tabs for multiple ingestion channels
tab1, tab2 = st.tabs(["🌐 Web Ingest (URL)", "📄 PDF Document Ingest"])

with tab1:
    st.subheader("Smart Web Crawler (Live Streaming)")
    target_url = st.text_input("Recipe or Category Webpage URL", placeholder="https://chitrasfoodbook.com/... or category link")

    if st.button("Process Web URL Live"):
        if not target_url:
            st.warning("Please enter a URL first!")
        else:
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
                            current = data.get("current")
                            total = data.get("total")
                            recipe = data.get("recipe")
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

with tab2:
    st.subheader("Upload Recipe PDF")
    uploaded_file = st.file_uploader("Choose a recipe PDF file", type=["pdf"])

    if uploaded_file is not None:
        if st.button("Upload & Parse PDF"):
            with st.spinner("Uploading PDF, extracting text with PyMuPDF, and saving to vault..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                    response = requests.post(
                        "http://127.0.0.1:8000/api/v1/ingest-pdf",
                        files=files
                    )
                    
                    if response.status_code == 200:
                        res_data = response.json()
                        st.success(f"Successfully ingested PDF: **{res_data.get('title')}**")
                        st.json(res_data)
                    else:
                        st.error(f"Server error: {response.text}")
                except Exception as e:
                    st.error(f"Failed to connect to backend: {e}")