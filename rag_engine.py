import os
import threading
import traceback
import logging
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
    global db, is_loading, load_error
    
    try:
        logger.info("📥 Starting vector store download...")
        
        # Create local directory
        os.makedirs(LOCAL_PATH, exist_ok=True)
        
        # Check for required environment variables
        if not os.getenv("SUPABASE_URL") or not os.getenv("SUPABASE_KEY"):
            logger.warning("⚠️ Supabase credentials not found. Skipping vector store load.")
            is_loading = False
            return
        
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
                logger.warning(f"⚠️ Failed to download {remote_path}, trying to continue...")
                # Don't crash if one file fails
                continue
        
        # Check if files exist
        faiss_file = os.path.join(LOCAL_PATH, "index.faiss")
        pkl_file = os.path.join(LOCAL_PATH, "index.pkl")
        
        if not os.path.exists(faiss_file) or not os.path.exists(pkl_file):
            logger.warning("⚠️ Vector store files not found. RAG will use fallback responses.")
            is_loading = False
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
        
        # Test the vector store
        test_results = db.similarity_search("test", k=1)
        logger.info(f"✅ Vector store loaded! Test returned {len(test_results)} results")
        
        is_loading = False
        logger.info("🎉 Vector store ready!")
        
    except Exception as e:
        logger.error(f"❌ Vector store loading failed: {str(e)}")
        load_error = str(e)
        is_loading = False
        # Don't crash the app, just log the error

def start_loading_vectorstore():
    """Start loading vector store in background thread"""
    thread = threading.Thread(target=load_vectorstore, daemon=True)
    thread.start()
    logger.info("🔄 Vector store loading in background...")

def get_answer(question: str, k: int = 4) -> str:
    """Get answer using RAG with Gemini or fallback"""
    global db, gemini_client
    
    # Handle greetings
    question_lower = question.lower().strip()
    greetings = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]
    
    if question_lower in greetings:
        return "Hello! 👋 I'm the Primis Digital AI assistant. How can I help you today?"
    
    try:
        # Check if system is ready
        if db is None:
            logger.warning("⚠️ Vector store not loaded yet, using fallback response")
            return "I'm still loading the knowledge base. Please try again in a moment or ask about Primis Digital services in the meantime."
        
        if gemini_client is None:
            logger.warning("⚠️ Gemini client not initialized")
            # Try to search vector store anyway
            docs = db.similarity_search(question, k=k)
            if docs:
                return f"I found information about '{question}' in our knowledge base. For detailed answers, the AI system needs to be fully initialized."
            return "The AI system is still initializing. Please try again shortly."
        
        # Search vector store
        logger.info(f"🔍 Searching for: {question}")
        docs = db.similarity_search(question, k=k)
        
        if not docs:
            logger.warning("⚠️ No relevant documents found")
            return "I couldn't find specific information about that in the Primis Digital knowledge base. Could you rephrase or ask about our services, technologies, or company information?"
        
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
        return "I'm having trouble generating a response right now. Please try again or ask a different question about Primis Digital."
