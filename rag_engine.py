import os
import threading
import traceback
import logging
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

# Global variables
db = None
is_loading = True
gemini_client = None
loading_error = None

def initialize_gemini():
    """Initialize Gemini client"""
    global gemini_client
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment")
        
        gemini_client = genai.Client(api_key=api_key)
        logger.info("✅ Gemini client initialized")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to initialize Gemini: {str(e)}")
        return False

def load_vectorstore():
    """Load vector store from Supabase with detailed error logging"""
    global db, is_loading, loading_error
    
    try:
        logger.info("=" * 70)
        logger.info("📥 STARTING VECTOR STORE DOWNLOAD")
        logger.info("=" * 70)
        
        # Log environment variables (without exposing secrets)
        logger.info(f"🔧 SUPABASE_URL: {os.getenv('SUPABASE_URL', 'NOT SET')[:30]}...")
        logger.info(f"🔧 SUPABASE_KEY: {'SET' if os.getenv('SUPABASE_KEY') else 'NOT SET'}")
        logger.info(f"🔧 BUCKET_NAME: {BUCKET_NAME}")
        logger.info(f"🔧 LOCAL_PATH: {LOCAL_PATH}")
        
        # Create local directory
        logger.info(f"📁 Creating directory: {LOCAL_PATH}")
        os.makedirs(LOCAL_PATH, exist_ok=True)
        
        # Initialize Supabase manager
        logger.info("🔗 Initializing Supabase manager...")
        storage = SupabaseStorageManager()
        logger.info("✅ Supabase manager initialized")
        
        # Download files from Supabase
        files_to_download = ["index.faiss", "index.pkl"]
        
        for filename in files_to_download:
            remote_path = f"{REMOTE_FOLDER}/{filename}"
            local_file = os.path.join(LOCAL_PATH, filename)
            
            logger.info(f"⬇️  Downloading {remote_path}...")
            success = storage.download_file(remote_path, local_file, BUCKET_NAME)
            
            if not success:
                raise Exception(f"Failed to download {remote_path}")
            
            # Verify file exists and has content
            if os.path.exists(local_file):
                size = os.path.getsize(local_file)
                logger.info(f"✅ Downloaded {filename}: {size:,} bytes")
                
                if size == 0:
                    raise Exception(f"{filename} is empty (0 bytes)")
            else:
                raise Exception(f"File not found after download: {filename}")
        
        # Initialize embeddings
        logger.info("🔧 Initializing embeddings model...")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        logger.info("✅ Embeddings model initialized")
        
        # Load FAISS index
        logger.info("📚 Loading FAISS index from local files...")
        db = FAISS.load_local(
            LOCAL_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
        logger.info("✅ FAISS index loaded successfully")
        
        # Test the vector store
        logger.info("🧪 Testing vector store with sample query...")
        test_results = db.similarity_search("test query", k=1)
        logger.info(f"✅ Test search returned {len(test_results)} results")
        
        if test_results:
            logger.info(f"📄 Sample content: {test_results[0].page_content[:100]}...")
        
        is_loading = False
        logger.info("=" * 70)
        logger.info("🎉 VECTOR STORE LOADED SUCCESSFULLY!")
        logger.info("=" * 70)
        
    except Exception as e:
        error_msg = f"Vector store loading failed: {str(e)}"
        logger.error("=" * 70)
        logger.error(f"❌ {error_msg}")
        logger.error("=" * 70)
        logger.error(traceback.format_exc())
        
        loading_error = error_msg
        is_loading = False  # Set to False so we stop saying "loading"
        db = None

def start_loading_vectorstore():
    """Start loading vector store in background thread"""
    thread = threading.Thread(target=load_vectorstore, daemon=True)
    thread.start()
    logger.info("🔄 Vector store loading started in background thread...")

def get_answer(question: str, k: int = 4) -> str:
    """Get answer using RAG with Gemini"""
    global db, gemini_client, loading_error
    
    try:
        # Check if there was a loading error
        if loading_error:
            return f"❌ System initialization failed: {loading_error}. Please check the logs or contact support."
        
        # Check if still loading
        if db is None and is_loading:
            return "I'm still loading the knowledge base. This usually takes 30-60 seconds after startup. Please try again in a moment."
        
        # Check if failed to load
        if db is None and not is_loading:
            return "❌ The knowledge base failed to load. Please contact support."
        
        # Check Gemini client
        if gemini_client is None:
            return "❌ Gemini AI client not initialized. Please contact support."
        
        # Search vector store
        logger.info(f"🔍 Searching for: {question}")
        docs = db.similarity_search(question, k=k)
        
        if not docs:
            logger.warning("⚠️ No relevant documents found")
            return "I couldn't find relevant information in the Primis Digital knowledge base. Could you rephrase your question?"
        
        # Log retrieved documents
        logger.info(f"📚 Found {len(docs)} relevant documents")
        for i, doc in enumerate(docs):
            logger.info(f"  Doc {i+1}: {doc.page_content[:100]}...")
        
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
