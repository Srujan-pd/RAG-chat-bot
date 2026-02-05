import os
import threading
import traceback
from supabase_manager import SupabaseStorageManager
# Updated import to match your requirements.txt
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Configuration
BUCKET_NAME = os.getenv("SUPABASE_BUCKET_NAME", "vectorstore-bucket")
REMOTE_FOLDER = "vectorstore"
LOCAL_PATH = "/tmp/vectorstore"

db = None
is_loading = True

def load_vectorstore():
    global db, is_loading
    try:
        print("🚀 [RAG THREAD] Starting background load...")
        
        # 1. Download files from Supabase
        sm = SupabaseStorageManager()
        print(f"📦 [RAG THREAD] Downloading index from {BUCKET_NAME}...")
        sm.download_vectorstore(BUCKET_NAME, REMOTE_FOLDER, LOCAL_PATH)
        
        # 2. Initialize Embeddings (FORCE /tmp for caching)
        print("🔧 [RAG THREAD] Initializing HuggingFace Embeddings...")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            cache_folder="/tmp/hf_cache"  # Mandatory for Cloud Run's read-only filesystem
        )
        
        # 3. Load FAISS from local /tmp path
        print(f"💾 [RAG THREAD] Loading FAISS index from {LOCAL_PATH}...")
        if os.path.exists(os.path.join(LOCAL_PATH, "index.faiss")):
            db = FAISS.load_local(
                LOCAL_PATH, 
                embeddings, 
                allow_dangerous_deserialization=True
            )
            is_loading = False
            print("✅ [RAG THREAD] RAG Engine Ready!")
        else:
            print(f"❌ [RAG THREAD] Error: index.faiss not found in {LOCAL_PATH}")

    except Exception as e:
        print(f"❌ [RAG THREAD] CRITICAL ERROR: {str(e)}")
        print(traceback.format_exc())

# Start the background thread immediately
thread = threading.Thread(target=load_vectorstore, daemon=True)
thread.start()
