import os
import threading
import traceback
import logging
import time
from supabase_manager import SupabaseStorageManager
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from google import genai
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
BUCKET_NAME = os.getenv("SUPABASE_BUCKET_NAME", "vectorstore-bucket")
REMOTE_FOLDER = "vectorstore"
LOCAL_PATH = "/tmp/vectorstore"

# Global variables
db = None
is_loading = False  # Start as False
gemini_client = None
load_error = None
load_complete = threading.Event()

def initialize_gemini():
    """Initialize Gemini client"""
    global gemini_client
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.error("❌ GEMINI_API_KEY not found in environment")
            return False
        
        gemini_client = genai.Client(api_key=api_key)
        logger.info("✅ Gemini client initialized")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to initialize Gemini: {str(e)}")
        return False

def load_vectorstore():
    """Load vector store from Supabase with better error handling"""
    global db, is_loading, load_error, load_complete
    
    is_loading = True
    load_complete.clear()
    
    try:
        logger.info("📥 Starting vector store download...")
        
        # Create local directory
        os.makedirs(LOCAL_PATH, exist_ok=True)
        
        # Check for required environment variables
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_KEY")
        
        if not supabase_url or not supabase_key:
            logger.error("❌ Supabase credentials not found")
            logger.error(f"SUPABASE_URL: {'Set' if supabase_url else 'Not set'}")
            logger.error(f"SUPABASE_KEY: {'Set' if supabase_key else 'Not set'}")
            load_error = "Supabase credentials missing"
            is_loading = False
            load_complete.set()
            return
        
        logger.info("🔗 Connecting to Supabase...")
        
        try:
            storage = SupabaseStorageManager()
            logger.info("✅ Supabase client initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Supabase client: {e}")
            load_error = f"Supabase init failed: {e}"
            is_loading = False
            load_complete.set()
            return
        
        # Download files from Supabase
        files_to_download = ["index.faiss", "index.pkl"]
        
        for filename in files_to_download:
            remote_path = f"{REMOTE_FOLDER}/{filename}"
            local_file = os.path.join(LOCAL_PATH, filename)
            
            logger.info(f"⬇️  Downloading {remote_path}...")
            try:
                success = storage.download_file(remote_path, local_file, BUCKET_NAME)
                
                if not success:
                    logger.error(f"❌ Failed to download {remote_path}")
                    load_error = f"Failed to download {remote_path}"
                    is_loading = False
                    load_complete.set()
                    return
                else:
                    size = os.path.getsize(local_file) if os.path.exists(local_file) else 0
                    logger.info(f"✅ Downloaded {filename}: {size:,} bytes")
            except Exception as e:
                logger.error(f"❌ Error downloading {remote_path}: {e}")
                load_error = f"Download error: {e}"
                is_loading = False
                load_complete.set()
                return
        
        # Check if files exist
        faiss_file = os.path.join(LOCAL_PATH, "index.faiss")
        pkl_file = os.path.join(LOCAL_PATH, "index.pkl")
        
        if not os.path.exists(faiss_file):
            logger.error(f"❌ FAISS file not found: {faiss_file}")
            load_error = "FAISS file not found after download"
            is_loading = False
            load_complete.set()
            return
        
        if not os.path.exists(pkl_file):
            logger.error(f"❌ PKL file not found: {pkl_file}")
            load_error = "PKL file not found after download"
            is_loading = False
            load_complete.set()
            return
        
        logger.info(f"✅ Files downloaded: FAISS={os.path.exists(faiss_file)}, PKL={os.path.exists(pkl_file)}")
        
        # Initialize embeddings
        logger.info("🔧 Initializing embeddings...")
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            logger.info("✅ Embeddings initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize embeddings: {e}")
            load_error = f"Embeddings error: {e}"
            is_loading = False
            load_complete.set()
            return
        
        # Load FAISS index
        logger.info("📚 Loading FAISS index...")
        try:
            db = FAISS.load_local(
                LOCAL_PATH,
                embeddings,
                allow_dangerous_deserialization=True
            )
            
            # Log success
            vector_count = db.index.ntotal if hasattr(db.index, 'ntotal') else "unknown"
            logger.info(f"✅ Vector store loaded successfully! Vectors: {vector_count}")
            
            # Set global db
            globals()['db'] = db
            is_loading = False
            load_complete.set()
            logger.info("🎉 Vector store ready!")
            
        except Exception as e:
            logger.error(f"❌ Failed to load FAISS index: {e}")
            logger.error(traceback.format_exc())
            load_error = f"FAISS load error: {e}"
            is_loading = False
            load_complete.set()
        
    except Exception as e:
        logger.error(f"❌ Vector store loading failed: {str(e)}")
        logger.error(traceback.format_exc())
        load_error = str(e)
        is_loading = False
        load_complete.set()

def start_loading_vectorstore():
    """Start loading vector store in background thread"""
    thread = threading.Thread(target=load_vectorstore, daemon=True)
    thread.start()
    logger.info("🔄 Vector store loading started in background...")

def wait_for_vectorstore(timeout=60):
    """Wait for vector store to load with timeout"""
    return load_complete.wait(timeout=timeout)

def get_answer(question: str, k: int = 4) -> str:
    """Get answer using RAG with Gemini"""
    global db, gemini_client, is_loading
    
    # Handle greetings
    question_lower = question.lower().strip()
    greetings = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening", "hi there", "hello there"]
    
    if question_lower in greetings:
        return "Hello! 👋 I'm the Primis Digital AI assistant. I can help you with information about our services, technologies, and company. How can I assist you today?"
    
    # Check if still loading
    if is_loading:
        logger.warning("⚠️ Vector store still loading when request received")
        return "I'm still loading the knowledge base. This usually takes 30-60 seconds after startup. Please try again in a moment."
    
    # Check if failed to load
    if db is None:
        logger.error(f"❌ Vector store is None. Load error: {load_error}")
        
        # Provide helpful error message
        if load_error:
            return f"I'm having trouble accessing the knowledge base. Error: {load_error}. The system will retry automatically."
        else:
            return "The knowledge base is not loaded yet. Please wait a moment and try again."
    
    if gemini_client is None:
        logger.error("❌ Gemini client is None")
        return "The AI service is not initialized. Please check if GEMINI_API_KEY is set correctly."
    
    try:
        # Search vector store
        logger.info(f"🔍 Searching for: '{question}'")
        docs = db.similarity_search(question, k=k)
        
        if not docs:
            logger.warning(f"⚠️ No relevant documents found for: '{question}'")
            return "I couldn't find specific information about that in our knowledge base. You can ask about: our services (AI development, DevOps, web applications), technologies we use, company information, or how to contact us."
        
        # Log retrieved documents
        logger.info(f"📚 Found {len(docs)} relevant documents")
        
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
- Be specific and cite relevant details from the context
- If the context doesn't contain enough information, politely say so and suggest related topics we can help with
- Keep your answer concise and professional (2-4 paragraphs maximum)
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
        return f"I encountered an error while processing your request. Please try again with a different question."
