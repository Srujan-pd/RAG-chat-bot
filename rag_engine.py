import os
import threading
import traceback
import time
from typing import Optional
from supabase_manager import SupabaseStorageManager
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from google import genai
from google.genai import types

# Configuration
BUCKET_NAME = os.getenv("SUPABASE_BUCKET_NAME", "vectorstore-bucket")
REMOTE_FOLDER = "vectorstore"
LOCAL_PATH = "/tmp/vectorstore"

# Global variables
db = None
is_loading = False
loading_error: Optional[str] = None
loading_complete = False
load_attempts = 0
MAX_LOAD_ATTEMPTS = 3

def download_vectorstore_with_retry():
    """Download vector store with retry logic"""
    global load_attempts
    
    sm = SupabaseStorageManager()
    
    for attempt in range(MAX_LOAD_ATTEMPTS):
        try:
            load_attempts = attempt + 1
            print(f"📦 Attempt {load_attempts}/{MAX_LOAD_ATTEMPTS}: Downloading vector store...")
            
            # Check if files exist in Supabase
            sm.download_vectorstore(BUCKET_NAME, REMOTE_FOLDER, LOCAL_PATH)
            
            # Verify files were downloaded
            faiss_path = os.path.join(LOCAL_PATH, "index.faiss")
            pkl_path = os.path.join(LOCAL_PATH, "index.pkl")
            
            if os.path.exists(faiss_path) and os.path.exists(pkl_path):
                print(f"✅ Vector store downloaded successfully")
                print(f"   - index.faiss: {os.path.getsize(faiss_path)} bytes")
                print(f"   - index.pkl: {os.path.getsize(pkl_path)} bytes")
                return True
            else:
                print(f"⚠️ Files not found after download")
                print(f"   Looking for: {faiss_path}, {pkl_path}")
                if attempt < MAX_LOAD_ATTEMPTS - 1:
                    time.sleep(2)  # Wait before retry
                    
        except Exception as e:
            print(f"❌ Download attempt {load_attempts} failed: {str(e)}")
            if attempt < MAX_LOAD_ATTEMPTS - 1:
                time.sleep(2)  # Wait before retry
    
    return False

def load_vectorstore():
    """Load vector store from Supabase in background thread"""
    global db, is_loading, loading_error, loading_complete
    
    is_loading = True
    try:
        print("🚀 [RAG THREAD] Starting background load...")
        print(f"📊 Configuration:")
        print(f"   - Bucket: {BUCKET_NAME}")
        print(f"   - Remote folder: {REMOTE_FOLDER}")
        print(f"   - Local path: {LOCAL_PATH}")
        
        # 1. Download vector store files with retry
        download_success = download_vectorstore_with_retry()
        
        if not download_success:
            error_msg = f"Failed to download vector store after {MAX_LOAD_ATTEMPTS} attempts"
            raise Exception(error_msg)
        
        # 2. Initialize HuggingFace Embeddings with LOCAL FILES ONLY
        print("🔧 [RAG THREAD] Initializing HuggingFace Embeddings...")
        
        # Set environment variables to force local files only
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        os.environ['HF_DATASETS_OFFLINE'] = '1'
        os.environ['HF_HUB_OFFLINE'] = '1'
        
        # Use local cache folder (/tmp) where the model was pre-downloaded during build
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            cache_folder="/tmp",  # Use /tmp where model was cached
            model_kwargs={
                'device': 'cpu',
                'local_files_only': True  # CRITICAL: Don't try to download
            },
            encode_kwargs={
                'normalize_embeddings': True
            }
        )
        print("✅ HuggingFace Embeddings initialized (offline mode)")
        
        # 3. Load FAISS from local /tmp path
        faiss_path = os.path.join(LOCAL_PATH, "index.faiss")
        pkl_path = os.path.join(LOCAL_PATH, "index.pkl")
        
        print(f"💾 [RAG THREAD] Loading FAISS index...")
        print(f"   - FAISS file exists: {os.path.exists(faiss_path)}")
        print(f"   - PKL file exists: {os.path.exists(pkl_path)}")
        
        if os.path.exists(faiss_path) and os.path.exists(pkl_path):
            db = FAISS.load_local(
                LOCAL_PATH, 
                embeddings, 
                allow_dangerous_deserialization=True
            )
            is_loading = False
            loading_complete = True
            print(f"✅ [RAG THREAD] RAG Engine Ready! Vector store loaded with {db.index.ntotal} vectors")
        else:
            error_msg = f"Vector store files not found after download: {faiss_path}, {pkl_path}"
            print(f"❌ [RAG THREAD] {error_msg}")
            raise Exception(error_msg)

    except Exception as e:
        error_msg = f"CRITICAL ERROR: {str(e)}"
        print(f"❌ [RAG THREAD] {error_msg}")
        print(traceback.format_exc())
        loading_error = error_msg
        loading_complete = True
        is_loading = False

def start_vectorstore_loading():
    """Start the vector store loading in background"""
    thread = threading.Thread(target=load_vectorstore, daemon=True)
    thread.start()
    print("✅ Vector store loading started in background thread")

def rag_answer(question: str) -> str:
    """
    Main RAG function to answer questions using vector store and Gemini
    """
    global db, is_loading, loading_error, loading_complete
    
    # Check if vector store is still loading
    if is_loading:
        return "⚠️ The knowledge base is still loading. Please try again in 10-15 seconds."
    
    # Check if vector store failed to load
    if loading_error:
        return f"⚠️ Knowledge base unavailable: {loading_error}"
    
    # Check if vector store is loaded
    if db is None:
        if loading_complete:
            return "⚠️ Knowledge base failed to load. Please check the system configuration."
        return "⚠️ Knowledge base not loaded yet. Please wait a moment and try again."
    
    try:
        # 1. Search for relevant context
        print(f"🔍 Searching for: '{question}'")
        docs = db.similarity_search(question, k=4)
        
        if not docs:
            return "I couldn't find specific information about that in our knowledge base. Could you rephrase your question?"
        
        print(f"✅ Found {len(docs)} relevant documents")
        
        # 2. Prepare context
        context_parts = []
        for i, doc in enumerate(docs, 1):
            # Extract metadata if available
            metadata = getattr(doc, 'metadata', {})
            source = metadata.get('source', metadata.get('url', 'Website content'))
            
            # Clean and truncate content
            content = doc.page_content.strip()
            if len(content) > 500:
                content = content[:500] + "..."
            
            context_parts.append(f"[Document {i} from {source}]\n{content}")
        
        context = "\n\n".join(context_parts)
        
        # 3. Prepare Gemini prompt
        prompt = f"""You are a helpful AI assistant for Primis Digital. Answer the user's question using ONLY the information provided below. 
If the information isn't in the provided context, say "I don't have specific information about that in our knowledge base."

Context from Primis Digital website:
{context}

Question: {question}

Answer (be concise and helpful, using only the context above):"""
        
        # 4. Call Gemini API
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return "⚠️ AI service is not configured properly."
        
        # Remove conflicting env var
        if "GOOGLE_API_KEY" in os.environ:
            del os.environ["GOOGLE_API_KEY"]
        
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=500
            )
        )
        
        answer = response.text.strip()
        print(f"🤖 Generated answer: {answer[:100]}...")
        return answer
        
    except Exception as e:
        print(f"❌ RAG Error: {str(e)}")
        print(traceback.format_exc())
        return f"Sorry, I encountered an error while processing your question: {str(e)}"

# Start vector store loading when module is imported
try:
    start_vectorstore_loading()
except Exception as e:
    print(f"❌ Failed to start vector store loading: {str(e)}")
    loading_error = str(e)
    loading_complete = True
