import os
import threading
import traceback
import logging
import time
from supabase_manager import SupabaseStorageManager
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from google import genai
from google.genai import types
from dotenv import load_dotenv
from models import Chat

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
is_loaded = False
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

def wait_for_vectorstore(timeout=45):
    """Wait for vector store to load with timeout"""
    global is_loading, db, loading_error
    
    start_time = time.time()
    while is_loading:
        if time.time() - start_time > timeout:
            logger.error(f"❌ Vector store loading timeout after {timeout}s")
            return False
        time.sleep(0.5)
    
    global is_loaded
    is_loaded = db is not None
    
    if loading_error:
        logger.error(f"❌ Vector store failed to load: {loading_error}")
        return False
        
    return is_loaded

def load_vectorstore():
    """Load vector store from Supabase"""
    global db, is_loading, loading_error

    try:
        logger.info("📥 Starting vector store download...")

        os.makedirs(LOCAL_PATH, exist_ok=True)
        storage = SupabaseStorageManager()

        files_to_download = ["index.faiss", "index.pkl"]

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
            else:
                raise Exception(f"File not found after download: {filename}")

        logger.info("🔧 Initializing embeddings...")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            cache_folder="/app/model_cache"
        )

        logger.info("📚 Loading FAISS index...")
        db = FAISS.load_local(
            LOCAL_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )

        # Test the vector store
        test_results = db.similarity_search("contact phone email", k=2)
        logger.info(f"✅ Vector store loaded! Test search returned {len(test_results)} results")

        if test_results:
            logger.info(f"📄 Sample content: {test_results[0].page_content[:200]}...")
            
            # Check if contact info is in the results
            contact_keywords = ['phone', 'email', 'contact', 'call', '📞', '✉️']
            for i, doc in enumerate(test_results):
                for keyword in contact_keywords:
                    if keyword.lower() in doc.page_content.lower():
                        logger.info(f"   ✅ Contact info found in doc {i+1}")
                        break

        is_loading = False
        loading_error = None
        logger.info("🎉 Vector store ready!")

    except Exception as e:
        logger.error(f"❌ Vector store loading failed: {str(e)}")
        logger.error(traceback.format_exc())
        loading_error = str(e)
        is_loading = False
        db = None

def start_loading_vectorstore():
    """Start loading vector store in background thread"""
    thread = threading.Thread(target=load_vectorstore, daemon=True)
    thread.start()
    logger.info("🔄 Vector store loading in background...")

def get_recent_messages(db_session, session_id, limit=5):
    """Fetch recent chat messages for context"""
    try:
        chats = (
            db_session.query(Chat)
            .filter(Chat.session_id == session_id)
            .order_by(Chat.created_at.desc())
            .limit(limit)
            .all()
        )
        return list(reversed(chats))
    except Exception as e:
        logger.error(f"❌ Error fetching recent messages: {e}")
        return []

def rewrite_question(chat_history, user_question):
    """Convert follow-up questions into standalone questions"""
    if not chat_history:
        return user_question
        
    try:
        conversation = ""
        for chat in chat_history[-3:]:  # Use last 3 exchanges
            conversation += f"User: {chat.question}\nAssistant: {chat.answer}\n"

        prompt = f"""
Given the conversation below and a follow-up question, rewrite the follow-up question 
so it can be understood by itself without the conversation context.

Conversation:
{conversation}

Follow-up question: {user_question}

Rewritten question:
"""
        response = gemini_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.1)
        )

        rewritten = response.text.strip()
        logger.info(f"🔄 Rewritten: '{user_question}' -> '{rewritten}'")
        return rewritten
    except Exception as e:
        logger.error(f"❌ Error rewriting question: {e}")
        return user_question

def get_answer(question, session_id=None, db_session=None):
    """Main function to get RAG-based answers"""
    global db, gemini_client
    
    # Check if vector store is loaded
    if db is None:
        logger.info("⏳ Vector store not loaded, waiting...")
        if not wait_for_vectorstore():
            return (
                "🔄 System is initializing. Please try again in a few seconds.\n\n"
                "If this persists, contact support@primisdigital.com"
            )
    
    try:
        # Prepare search query
        search_query = question
        
        # Rewrite question if we have chat history
        if session_id and db_session:
            chat_history = get_recent_messages(db_session, session_id)
            if chat_history:
                search_query = rewrite_question(chat_history, question)
        
        # Search for relevant documents
        logger.info(f"🔍 Searching for: {search_query}")
        docs = db.similarity_search(search_query, k=6)  # Increased from 4
        
        if not docs:
            logger.warning("⚠️ No relevant documents found")
            return (
                "I couldn't find specific information about that in our knowledge base. "
                "However, you can:\n\n"
                "📧 Email us at: contact@primisdigital.com\n"
                "🌐 Visit our website: https://primisdigital.com\n"
                "💬 Use the contact form on our site\n\n"
                "What specific information are you looking for?"
            )

        # Log found documents
        logger.info(f"📚 Found {len(docs)} relevant documents")
        
        # Build context with source URLs if available
        context_parts = []
        for i, doc in enumerate(docs):
            # Try to extract URL if present in content
            content = doc.page_content
            context_parts.append(f"[Document {i+1}]\n{content}")
        
        context = "\n\n---\n\n".join(context_parts)

        # Enhanced prompt that ALWAYS tries to help
        prompt = f"""You are Primis Digital's official support AI assistant. Your goal is to ALWAYS be helpful.

CONTEXT FROM PRIMIS DIGITAL WEBSITE:
{context}

USER QUESTION: {question}

CRITICAL INSTRUCTIONS - FOLLOW THESE EXACTLY:

1. FIRST, search the context for ANY contact information (phone, email, address, contact form)
   IF you find contact information, ALWAYS include it in your response

2. If the context contains phone numbers: Format as 📞 Phone: [number]
3. If the context contains emails: Format as ✉️ Email: [email]
4. If the context contains addresses: Format as 📍 Address: [address]

5. If the user asks for contact info and you CANNOT find it in the context, STILL provide:
   "Based on our website, here's how to reach Primis Digital:
    📧 General inquiries: contact@primisdigital.com
    🌐 Website: https://primisdigital.com
    📝 Contact form: Available on our website
   
    For specific departments, please let me know what you need help with."

6. NEVER say "I don't have that information" without providing alternatives
7. ALWAYS be specific and cite details from the context
8. Keep responses professional, clear, and well-formatted

YOUR RESPONSE:
"""
        
        logger.info("🤖 Generating answer with Gemini...")
        response = gemini_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=600,
                top_p=0.95
            )
        )

        answer = response.text
        logger.info(f"✅ Answer generated: {len(answer)} chars")
        
        return answer

    except Exception as e:
        logger.error(f"❌ Error in get_answer: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Fallback response with contact info
        return (
            "I'm experiencing a technical issue at the moment. "
            "For immediate assistance, please contact us directly:\n\n"
            "📧 Email: contact@primisdigital.com\n"
            "🌐 Website: https://primisdigital.com\n\n"
            "Please try again in a few moments."
        )
