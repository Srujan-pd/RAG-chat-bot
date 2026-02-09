import os
import threading
import traceback
import logging
import time
from functools import lru_cache
from supabase_manager import SupabaseStorageManager
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from google import genai

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
BUCKET_NAME = os.getenv("SUPABASE_BUCKET_NAME", "vectorstore-bucket")
REMOTE_FOLDER = "vectorstore"
LOCAL_PATH = "/tmp/vectorstore"
LOCAL_VECTORSTORE_PATH = "vectorstore"

# Global variables
vectorstore = None
gemini_client = None
init_error = None
is_rag_initialized = False

# Internal variables
_db = None
_is_loading = False
_loading_error = None
_last_load_attempt = None
_load_successful = False
_load_lock = threading.Lock()  # Add lock to prevent concurrent loads

def initialize_gemini():
    """Initialize Gemini client"""
    global gemini_client
    try:
        # Remove conflicting environment variable if exists
        if "GOOGLE_API_KEY" in os.environ:
            logger.info("Removing GOOGLE_API_KEY from environment...")
            del os.environ["GOOGLE_API_KEY"]
        
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment")
        
        logger.info("Initializing Gemini client...")
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
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        logger.info("✅ Embeddings model loaded")
        return embeddings
    except Exception as e:
        logger.error(f"❌ Failed to load embeddings model: {str(e)}")
        raise

def try_load_from_supabase():
    """Try to load vector store from Supabase"""
    try:
        logger.info("📥 Attempting to load vector store from Supabase...")
        
        # Create local directory
        os.makedirs(LOCAL_PATH, exist_ok=True)
        logger.info(f"Created directory: {LOCAL_PATH}")
        
        # Initialize Supabase manager
        logger.info("Initializing Supabase manager...")
        storage = SupabaseStorageManager()
        logger.info("✅ Supabase manager initialized")
        
        # Download files from Supabase
        files_to_download = ["index.faiss", "index.pkl"]
        downloaded_files = []
        
        for filename in files_to_download:
            remote_path = f"{REMOTE_FOLDER}/{filename}"
            local_file = os.path.join(LOCAL_PATH, filename)
            
            logger.info(f"⬇️  Downloading {remote_path}...")
            success = storage.download_file(remote_path, local_file, BUCKET_NAME)
            if not success:
                raise Exception(f"Failed to download {remote_path}")
            
            if os.path.exists(local_file):
                size = os.path.getsize(local_file)
                logger.info(f"✅ Downloaded {filename}: {size:,} bytes")
                downloaded_files.append(local_file)
            else:
                raise Exception(f"File not found after download: {filename}")
        
        # Load FAISS index
        logger.info("Loading FAISS index...")
        embeddings = get_embeddings()
        db = FAISS.load_local(
            LOCAL_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
        
        logger.info(f"✅ Vector store loaded from Supabase: {len(downloaded_files)} files")
        return db
        
    except Exception as e:
        logger.error(f"❌ Failed to load from Supabase: {str(e)}")
        return None

def try_load_from_local():
    """Try to load vector store from local bundled files"""
    try:
        logger.info("📁 Attempting to load vector store from local files...")
        
        # Check if local vectorstore directory exists
        if not os.path.exists(LOCAL_VECTORSTORE_PATH):
            logger.error(f"Local vectorstore directory not found: {LOCAL_VECTORSTORE_PATH}")
            return None
        
        # Check for required files
        required_files = ["index.faiss", "index.pkl"]
        for filename in required_files:
            file_path = os.path.join(LOCAL_VECTORSTORE_PATH, filename)
            if not os.path.exists(file_path):
                logger.error(f"Required file not found: {file_path}")
                return None
            size = os.path.getsize(file_path)
            logger.info(f"Found {filename}: {size:,} bytes")
        
        # Load FAISS index
        logger.info("Loading FAISS index from local files...")
        embeddings = get_embeddings()
        db = FAISS.load_local(
            LOCAL_VECTORSTORE_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
        
        logger.info("✅ Vector store loaded from local files")
        return db
        
    except Exception as e:
        logger.error(f"❌ Failed to load from local files: {str(e)}")
        return None

def load_vectorstore():
    """Load vector store with fallback strategy - BLOCKING"""
    global _db, _loading_error, vectorstore, _load_successful, _last_load_attempt, _is_loading
    
    # Use lock to prevent concurrent loading
    with _load_lock:
        # Check if already loaded
        if _db is not None:
            return True
        
        # Check if already loading
        if _is_loading:
            logger.info("Vector store is already being loaded by another thread")
            return False
        
        _is_loading = True
        _last_load_attempt = time.time()
    
    try:
        # Try Supabase first
        db = try_load_from_supabase()
        
        # If Supabase fails, try local files
        if db is None:
            logger.info("🔄 Falling back to local vector store...")
            db = try_load_from_local()
        
        if db is None:
            raise Exception("Failed to load vector store from any source")
        
        _db = db
        vectorstore = _db
        _load_successful = True
        _loading_error = None
        
        # Test the vector store
        test_results = _db.similarity_search("test", k=1)
        logger.info(f"🧪 Vector store test: Found {len(test_results)} results")
        
        return True
        
    except Exception as e:
        error_msg = f"Vector store loading failed: {str(e)}"
        logger.error(f"❌ {error_msg}")
        logger.error(traceback.format_exc())
        _loading_error = error_msg
        _load_successful = False
        return False
    finally:
        _is_loading = False

def init_rag():
    """Initialize RAG system - EAGER LOADING"""
    global is_rag_initialized, init_error
    
    try:
        logger.info("🧠 Initializing RAG system...")
        
        # Step 1: Initialize Gemini
        logger.info("Step 1: Initializing Gemini...")
        if not initialize_gemini():
            raise Exception("Failed to initialize Gemini client")
        
        # Step 2: Load vector store SYNCHRONOUSLY (not in background)
        logger.info("Step 2: Loading vector store...")
        start_time = time.time()
        
        if not load_vectorstore():
            raise Exception("Failed to load vector store")
        
        load_time = time.time() - start_time
        logger.info(f"✅ Vector store loaded in {load_time:.2f} seconds")
        
        # Mark as initialized
        is_rag_initialized = True
        logger.info("🎉 RAG initialization completed successfully")
        return True
        
    except Exception as e:
        error_msg = f"RAG initialization failed: {str(e)}"
        logger.error(f"❌ {error_msg}")
        logger.error(traceback.format_exc())
        init_error = error_msg
        is_rag_initialized = False
        return False

def get_answer(question: str, k: int = 4) -> str:
    """Get answer using RAG with Gemini"""
    global _db, gemini_client, _loading_error
    
    try:
        # Check Gemini client
        if gemini_client is None:
            logger.error("Gemini client not initialized")
            return "❌ AI service unavailable. Please try again later or contact support."
        
        # Check vector store
        if _db is None:
            logger.error("Vector store not loaded")
            if _loading_error:
                logger.error(f"Vector store error: {_loading_error}")
                return "❌ Knowledge base unavailable. Please contact support."
            return "❌ Knowledge base is still initializing. Please try again in a moment."
        
        # Search vector store
        logger.info(f"🔍 Searching for: {question}")
        docs = _db.similarity_search(question, k=k)
        
        if not docs:
            logger.warning("⚠️ No relevant documents found")
            return "I couldn't find relevant information in the Primis Digital knowledge base. Could you rephrase your question?"
        
        # Log retrieved documents
        logger.info(f"📚 Found {len(docs)} relevant documents")
        for i, doc in enumerate(docs):
            logger.debug(f"  Doc {i+1}: {doc.page_content[:100]}...")
        
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
        return f"❌ Error generating answer: {str(e)}"

