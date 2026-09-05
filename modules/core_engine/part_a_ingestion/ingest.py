import os
import json
from pathlib import Path
from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup

# Robust path resolution: go up 3 levels to the CognitiveKitchen root
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
STORAGE_DIR = Path(ROOT_DIR) / "data" / "raw_recipes"
STORAGE_DIR.mkdir(parents=True, exist_ok=True)

def scrape_and_store_recipe(url: str) -> dict:
    """
    Scrapes a recipe URL, strips website clutter, extracts clean markdown-like text for RAG,
    and saves it locally as a JSON document in our raw vault.
    """
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url, wait_until="domcontentloaded")
        html_content = page.content()
        page_title = page.title()
        browser.close()
        
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Strip away website clutter (ads, sidebars, navigation, footers)
    for element in soup(["script", "style", "nav", "footer", "header", "aside"]):
        element.decompose()
        
    # Extract clean, human-readable text body
    clean_text = soup.get_text(separator="\n", strip=True)
    
    recipe_document = {
        "url": url,
        "title": page_title,
        "raw_text": clean_text
    }
    
    # Generate a safe local filename from the page title
    safe_filename = "".join(c if c.isalnum() else "_" for c in page_title)[:50] + ".json"
    file_path = STORAGE_DIR / safe_filename
    
    # Save locally to our Raw Vault
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(recipe_document, f, indent=4, ensure_ascii=False)
        
    return {
        "status": "success", 
        "saved_path": str(file_path), 
        "title": page_title
    }