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
is_loading = True
gemini_client = None
load_error = None
load_complete = threading.Event()

def initialize_gemini():
    """Initialize Gemini client"""
    global gemini_client
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.warning("⚠️ GEMINI_API_KEY not found in environment")
            return False
        
        gemini_client = genai.Client(api_key=api_key)
        logger.info("✅ Gemini client initialized")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to initialize Gemini: {str(e)}")
        return False

def load_vectorstore():
    """Load vector store from Supabase"""
    global db, is_loading, load_error, load_complete
    
    try:
        logger.info("📥 Starting vector store download...")
        
        # Create local directory
        os.makedirs(LOCAL_PATH, exist_ok=True)
        
        # Check for required environment variables
        if not os.getenv("SUPABASE_URL") or not os.getenv("SUPABASE_KEY"):
            logger.warning("⚠️ Supabase credentials not found. Skipping vector store load.")
            is_loading = False
            load_complete.set()
            return
        
        # Initialize Supabase manager
        logger.info("🔗 Connecting to Supabase...")
        storage = SupabaseStorageManager()
        
        # Download files from Supabase
        files_to_download = ["index.faiss", "index.pkl"]
        
        for filename in files_to_download:
            remote_path = f"{REMOTE_FOLDER}/{filename}"
            local_file = os.path.join(LOCAL_PATH, filename)
            
            logger.info(f"⬇️  Downloading {remote_path}...")
            success = storage.download_file(remote_path, local_file, BUCKET_NAME)
            
            if not success:
                logger.error(f"❌ Failed to download {remote_path}")
                load_error = f"Failed to download {remote_path}"
                is_loading = False
                load_complete.set()
                return
        
        # Check if files exist
        faiss_file = os.path.join(LOCAL_PATH, "index.faiss")
        pkl_file = os.path.join(LOCAL_PATH, "index.pkl")
        
        if not os.path.exists(faiss_file) or not os.path.exists(pkl_file):
            logger.error("❌ Vector store files not found after download")
            load_error = "Vector store files not found"
            is_loading = False
            load_complete.set()
            return
        
        # Initialize embeddings
        logger.info("🔧 Initializing embeddings...")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Load FAISS index
        logger.info("📚 Loading FAISS index...")
        db = FAISS.load_local(
            LOCAL_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
        
        # Log success
        vector_count = db.index.ntotal if hasattr(db.index, 'ntotal') else "unknown"
        logger.info(f"✅ [RAG THREAD] RAG Engine Ready! Vector store loaded with {vector_count} vectors")
        
        is_loading = False
        load_complete.set()
        
    except Exception as e:
        logger.error(f"❌ Vector store loading failed: {str(e)}")
        load_error = str(e)
        is_loading = False
        load_complete.set()
        traceback.print_exc()

def start_loading_vectorstore():
    """Start loading vector store in background thread"""
    thread = threading.Thread(target=load_vectorstore, daemon=True)
    thread.start()
    logger.info("🔄 Vector store loading in background...")

def wait_for_vectorstore(timeout=60):
    """Wait for vector store to load with timeout"""
    logger.info(f"⏳ Waiting for vector store (timeout: {timeout}s)...")
    
    if load_complete.wait(timeout=timeout):
        if db is not None:
            logger.info("✅ Vector store loaded successfully")
            return True
        else:
            logger.warning("⚠️ Vector store loading completed but db is None")
            return False
    else:
        logger.warning(f"⚠️ Vector store loading timeout after {timeout} seconds")
        return False

def get_answer(question: str, k: int = 4) -> str:
    """Get answer using RAG with Gemini"""
    global db, gemini_client
    
    # Handle greetings immediately
    question_lower = question.lower().strip()
    greetings = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening", "hi there", "hello there"]
    
    if question_lower in greetings:
        return "Hello! 👋 I'm the Primis Digital AI assistant. I can help you with information about our services, technologies, and company. How can I assist you today?"
    
    # Check if still loading
    if is_loading:
        return "I'm still loading the knowledge base. Please try again in a moment. For now, you can ask about Primis Digital's services or contact information."
    
    # Check if failed to load
    if db is None:
        return "I'm having trouble accessing the knowledge base right now. Please try again shortly or contact support for assistance."
    
    if gemini_client is None:
        return "The AI service is temporarily unavailable. Please try again in a moment."
    
    try:
        # Search vector store
        logger.info(f"🔍 Searching for: {question}")
        docs = db.similarity_search(question, k=k)
        
        if not docs:
            logger.warning(f"⚠️ No relevant documents found for: {question}")
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
- If the context doesn't contain enough information, politely say so and suggest related topics
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
        return "I encountered an error while generating a response. Please try again with a different question or rephrase your query."
