import os
import threading
import traceback
import logging
from functools import lru_cache
from supabase_manager import SupabaseStorageManager
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from google import genai

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
BUCKET_NAME = os.getenv("SUPABASE_BUCKET_NAME", "vectorstore-bucket")
REMOTE_FOLDER = "vectorstore"
LOCAL_PATH = "/tmp/vectorstore"

# Global variables for external access (needed by main.py)
vectorstore = None
gemini_client = None
init_error = None
is_rag_initialized = False

# Internal variables
_db = None
_is_loading = False
_loading_error = None

def initialize_gemini():
    """Initialize Gemini client"""
    global gemini_client
    try:
        # Remove conflicting environment variable if exists
        if "GOOGLE_API_KEY" in os.environ:
            del os.environ["GOOGLE_API_KEY"]
        
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment")
        
        gemini_client = genai.Client(api_key=api_key)
        logger.info("✅ Gemini client initialized")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to initialize Gemini: {str(e)}")
        return False

@lru_cache(maxsize=1)
def get_embeddings():
    """Get cached embeddings model"""
    logger.info("🔧 Loading embeddings model...")
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

def load_vectorstore_sync():
    """Synchronously load vector store"""
    global _db, _loading_error, vectorstore
    
    try:
        logger.info("📥 Loading vector store...")
        
        # Create local directory
        os.makedirs(LOCAL_PATH, exist_ok=True)
        
        # Initialize Supabase manager
        storage = SupabaseStorageManager()
        
        # Download files from Supabase
        files_to_download = ["index.faiss", "index.pkl"]
        
        for filename in files_to_download:
            remote_path = f"{REMOTE_FOLDER}/{filename}"
            local_file = os.path.join(LOCAL_PATH, filename)
            
            logger.info(f"⬇️  Downloading {remote_path}...")
            success = storage.download_file(remote_path, local_file, BUCKET_NAME)
            if not success:
                raise Exception(f"Failed to download {remote_path}")
            
            if not os.path.exists(local_file) or os.path.getsize(local_file) == 0:
                raise Exception(f"File {filename} is empty or missing")
        
        # Load FAISS index
        embeddings = get_embeddings()
        _db = FAISS.load_local(
            LOCAL_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
        vectorstore = _db  # Update the public variable
        logger.info("✅ Vector store loaded successfully")
        return True
        
    except Exception as e:
        error_msg = f"Vector store loading failed: {str(e)}"
        logger.error(f"❌ {error_msg}")
        logger.error(traceback.format_exc())
        _loading_error = error_msg
        init_error = error_msg  # Update the public variable
        return False

def load_vectorstore_async():
    """Load vector store in background thread"""
    global _is_loading, is_rag_initialized
    
    def load_in_background():
        global _is_loading, is_rag_initialized
        _is_loading = True
        try:
            load_vectorstore_sync()
            is_rag_initialized = True
        except:
            pass
        finally:
            _is_loading = False
    
    thread = threading.Thread(target=load_in_background, daemon=True)
    thread.start()
    logger.info("🔄 Vector store loading started in background thread...")
    return thread

def ensure_vectorstore_loaded():
    """Ensure vector store is loaded (lazy loading)"""
    global _db, _is_loading, _loading_error
    
    if _db is not None:
        return True
    
    if _is_loading:
        logger.info("Vector store is still loading...")
        return False
        
    if _loading_error:
        return False
    
    # Try to load synchronously
    return load_vectorstore_sync()

def init_rag():
    """Initialize RAG system (called from main.py)"""
    global is_rag_initialized, init_error
    
    try:
        logger.info("🧠 Initializing RAG system...")
        
        # Step 1: Initialize Gemini
        if not initialize_gemini():
            raise Exception("Failed to initialize Gemini client")
        
        # Step 2: Start vector store loading in background
        load_vectorstore_async()
        
        # Mark as initialized (even though loading continues in background)
        is_rag_initialized = True
        logger.info("🎉 RAG initialization started successfully")
        
    except Exception as e:
        error_msg = f"RAG initialization failed: {str(e)}"
        logger.error(f"❌ {error_msg}")
        init_error = error_msg
        raise

def get_answer(question: str, k: int = 4) -> str:
    """Get answer using RAG with Gemini"""
    global _db, gemini_client, _loading_error
    
    try:
        # Check Gemini client
        if gemini_client is None and not initialize_gemini():
            return "❌ Gemini AI client not initialized. Please contact support."
        
        # Try to load vector store if not loaded
        if _db is None:
            if not ensure_vectorstore_loaded():
                if _loading_error:
                    return f"❌ Knowledge base unavailable: {_loading_error}"
                return "I'm still loading the knowledge base. Please try again in a moment."
        
        # Search vector store
        logger.info(f"🔍 Searching for: {question}")
        docs = _db.similarity_search(question, k=k)
        
        if not docs:
            logger.warning("⚠️ No relevant documents found")
            return "I couldn't find relevant information in the Primis Digital knowledge base. Could you rephrase your question?"
        
        # Combine context
        context = "\n\n---\n\n".join([doc.page_content for doc in docs])
        
        # Create prompt
        prompt = f"""You are a helpful assistant for Primis Digital, a technology company.

Based on the following information from Primis Digital's website, answer the user's question accurately and professionally.

CONTEXT FROM PRIMIS DIGITAL:
{context}

USER QUESTION: {question}

INSTRUCTIONS:
- Answer based ONLY on the provided context
- Be specific and cite relevant details
- If the context doesn't contain enough information, say so politely
- Keep your answer concise and professional
- Format your answer with clear paragraphs

ANSWER:"""

        # Get response from Gemini
        logger.info("🤖 Generating answer with Gemini...")
        response = gemini_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        
        answer = response.text
        logger.info(f"✅ Answer generated: {len(answer)} characters")
        
        return answer
        
    except Exception as e:
        logger.error(f"❌ Error in get_answer: {str(e)}")
        logger.error(traceback.format_exc())
        return f"Error generating answer: {str(e)}"

# For backward compatibility
def start_loading_vectorstore():
    """Alias for backward compatibility"""
    return load_vectorstore_async()
