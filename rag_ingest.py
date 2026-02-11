import requests
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from urllib.parse import urljoin, urlparse
import time
import os
import json
import logging
from supabase_manager import SupabaseStorageManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        }
        r = requests.get(url, timeout=15, headers=headers)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")

        # Remove unwanted tags
        for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
            tag.decompose()

        # Get text with better formatting
        text = soup.get_text(separator="\n", strip=True)
        
        # Clean up whitespace
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        text = ' '.join(lines)
        
        return text
    except Exception as e:
        logger.error(f"❌ Error fetching {url}: {e}")
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
            if href.startswith('#') or href.startswith('javascript:'):
                continue
                
            full_url = urljoin(base_url, href)
            full_url = full_url.split('#')[0].split('?')[0]

            if is_valid_url(full_url, base_url):
                links.add(full_url)

        return links
    except Exception as e:
        logger.error(f"❌ Error getting links from {url}: {e}")
        return set()

def crawl_website(start_url, max_pages=MAX_PAGES):
    """Crawl entire website starting from start_url"""
    visited = set()
    to_visit = {start_url}
    all_pages = []

    logger.info(f"🚀 Starting crawl from: {start_url}")

    while to_visit and len(visited) < max_pages:
        url = to_visit.pop()
        if url in visited:
            continue

        logger.info(f"🔹 [{len(visited) + 1}/{max_pages}] Crawling: {url}")

        text = fetch_text(url)
        if text and len(text) > 200:  # Only save pages with substantial content
            page_data = {
                "url": url,
                "content": text,
                "timestamp": time.time()
            }
            all_pages.append(page_data)
            logger.info(f"   ✅ Saved: {len(text)} chars")

        new_links = get_all_links(url, start_url)
        to_visit.update(new_links - visited)
        visited.add(url)
        time.sleep(DELAY)

    logger.info(f"✅ Crawl complete: {len(all_pages)} pages saved")
    return all_pages

def ingest_website():
    """Main function to crawl website and create vector store"""
    
    # Step 1: Crawl website
    all_pages = crawl_website(BASE_URL, MAX_PAGES)

    if not all_pages:
        logger.error("❌ No content found!")
        return

    # Save raw crawled data
    os.makedirs('data', exist_ok=True)
    with open('data/raw_crawl.json', 'w', encoding='utf-8') as f:
        json.dump(all_pages, f, ensure_ascii=False, indent=2)

    # Step 2: Combine and Chunk Text
    full_text = "\n\n".join([f"--- PAGE: {p['url']} ---\n\n{p['content']}" for p in all_pages])
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_text(full_text)
    logger.info(f"📄 Total chunks created: {len(chunks)}")

    # Save chunks
    with open('data/chunks_raw.json', 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    # Step 3: Create embeddings
    logger.info("🔧 Creating embeddings (HuggingFace)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Step 4: Create and Save FAISS vector store locally
    logger.info("💾 Saving FAISS index to local folder 'vectorstore'...")
    db = FAISS.from_texts(chunks, embeddings)
    db.save_local("vectorstore")

    # Step 5: Upload to Supabase
    logger.info("\n☁️ Connecting to Supabase...")
    try:
        storage = SupabaseStorageManager()
        
        logger.info(f"☁️ Uploading to Supabase bucket: {BUCKET_NAME}...")
        
        # Ensure the bucket exists
        storage.upload_file("vectorstore/index.faiss", "vectorstore/index.faiss", BUCKET_NAME)
        storage.upload_file("vectorstore/index.pkl", "vectorstore/index.pkl", BUCKET_NAME)
        
        logger.info("\n✅ Success! Website ingested and Supabase vector store updated.")
        
        # Verify upload
        files = storage.list_files(BUCKET_NAME, "vectorstore")
        logger.info(f"📁 Files in bucket: {len(files)}")
        
    except Exception as e:
        logger.error(f"\n❌ Supabase Upload Failed: {e}")
        logger.error("Check if 'vectorstore-bucket' exists in your Supabase Storage dashboard.")

if __name__ == "__main__":
    ingest_website()
