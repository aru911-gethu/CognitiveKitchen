import json
import time
from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
from backend.ingest import scrape_and_store_recipe

def stream_smart_ingest(url: str):
    """
    Generator that yields live progress updates for FastAPI streaming.
    """
    yield json.dumps({"status": "info", "message": f"Analyzing URL: {url}"}) + "\n"
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url, wait_until="domcontentloaded")
        html_content = page.content()
        browser.close()
        
    soup = BeautifulSoup(html_content, 'html.parser')
    
    recipe_links = set()
    for a in soup.find_all('a', href=True):
        href = a['href']
        if "/chitrasfoodbook.com/" in href:
            unwanted_patterns = [
                "/recipes", "/category/", "/tag/", "/page/", "wp-content", "#", 
                "/about-me", "/contact", "/recipe-index", "/privacy-policy", 
                "/terms", "/sitemap", "feed/"
            ]
            if not any(pattern in href for pattern in unwanted_patterns) and href != url and href != "https://chitrasfoodbook.com/":
                recipe_links.add(href)
                
    target_urls = list(recipe_links)
    
    if len(target_urls) > 1:
        yield json.dumps({"status": "info", "message": f"Found {len(target_urls)} pure recipe links. Scraping live..."}) + "\n"
        
        for idx, recipe_url in enumerate(target_urls, 1):
            try:
                res = scrape_and_store_recipe(recipe_url)
                yield json.dumps({
                    "status": "progress",
                    "current": idx,
                    "total": len(target_urls),
                    "recipe": res
                }) + "\n"
                time.sleep(1)
            except Exception as e:
                yield json.dumps({"status": "error", "message": f"Failed {recipe_url}: {str(e)}"}) + "\n"
        
        yield json.dumps({"status": "complete", "total_scraped": len(target_urls)}) + "\n"
    else:
        res = scrape_and_store_recipe(url)
        yield json.dumps({
            "status": "complete",
            "total_scraped": 1,
            "recipe": res
        }) + "\n"