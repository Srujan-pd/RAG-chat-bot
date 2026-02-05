import requests
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from urllib.parse import urljoin, urlparse
import time
import os
import json
from supabase_manager import SupabaseStorageManager

# Configuration
BASE_URL = "https://primisdigital.com/"
MAX_PAGES = 100
DELAY = 1
BUCKET_NAME = "vectorstore-bucket"

def is_valid_url(url, base_domain):
    """Check if URL belongs to the same domain"""
    parsed = urlparse(url)
    base_parsed = urlparse(base_domain)
    return parsed.netloc == base_parsed.netloc

def fetch_text(url):
    """Extract text content from a URL using BeautifulSoup"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        r = requests.get(url, timeout=10, headers=headers)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")

        # Remove unwanted tags
        for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
            tag.decompose()

        return soup.get_text(separator=" ", strip=True)
    except Exception as e:
        print(f"❌ Error fetching {url}: {e}")
        return ""

def get_all_links(url, base_url):
    """Extract all internal links from a page"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        r = requests.get(url, timeout=10, headers=headers)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")

        links = set()
        for link in soup.find_all('a', href=True):
            href = link['href']
            full_url = urljoin(base_url, href)
            full_url = full_url.split('#')[0].split('?')[0]

            if is_valid_url(full_url, base_url):
                links.add(full_url)

        return links
    except Exception as e:
        print(f"❌ Error getting links from {url}: {e}")
        return set()

def crawl_website(start_url, max_pages=MAX_PAGES):
    """Crawl entire website starting from start_url"""
    visited = set()
    to_visit = {start_url}
    all_pages = []

    print(f"🚀 Starting crawl from: {start_url}")

    while to_visit and len(visited) < max_pages:
        url = to_visit.pop()
        if url in visited:
            continue

        print(f"🔹 [{len(visited) + 1}/{max_pages}] Crawling: {url}")

        text = fetch_text(url)
        if text:
            all_pages.append({
                "url": url,
                "content": text
            })

        new_links = get_all_links(url, start_url)
        to_visit.update(new_links - visited)
        visited.add(url)
        time.sleep(DELAY)

    return all_pages

def ingest_website():
    """Main function to crawl website and create vector store"""
    
    # Step 1: Crawl website
    all_pages = crawl_website(BASE_URL, MAX_PAGES)

    if not all_pages:
        print("❌ No content found!")
        return

    # Step 2: Combine and Chunk Text
    full_text = "\n\n".join([f"--- PAGE: {p['url']} ---\n\n{p['content']}" for p in all_pages])
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_text(full_text)
    print(f"\n📄 Total chunks created: {len(chunks)}")

    # Step 3: Create embeddings
    print("🔧 Creating embeddings (HuggingFace)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Step 4: Create and Save FAISS vector store locally
    print("💾 Saving FAISS index to local folder 'vectorstore'...")
    db = FAISS.from_texts(chunks, embeddings)
    db.save_local("vectorstore")

    # Step 5: Upload to Supabase (The New Part)
    print("\n☁️ Connecting to Supabase...")
    try:
        storage = SupabaseStorageManager()
        
        print(f"☁️ Uploading to Supabase bucket: {BUCKET_NAME}...")
        # Uploading to the 'vectorstore' folder inside the bucket
        storage.upload_file("vectorstore/index.faiss", "vectorstore/index.faiss", BUCKET_NAME)
        storage.upload_file("vectorstore/index.pkl", "vectorstore/index.pkl", BUCKET_NAME)
        
        print("\n✅ Success! Website ingested and Supabase vector store updated.")
    except Exception as e:
        print(f"\n❌ Supabase Upload Failed: {e}")
        print("Check if 'vectorstore-bucket' exists in your Supabase Storage dashboard.")

if __name__ == "__main__":
    ingest_website()
