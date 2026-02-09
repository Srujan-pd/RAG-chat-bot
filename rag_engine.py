import os
import threading
import traceback
import logging
from supabase_manager import SupabaseStorageManager
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from google import genai
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
gemini_client = None


def initialize_gemini():
    """Initialize Gemini client"""
    global gemini_client
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.warning("GEMINI_API_KEY not found in environment")
            return False

        # Remove conflicting environment variable
        if "GOOGLE_API_KEY" in os.environ:
            del os.environ["GOOGLE_API_KEY"]
            
        gemini_client = genai.Client(api_key=api_key)
        logger.info("✅ Gemini client initialized")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to initialize Gemini: {str(e)}")
        return False


def load_vectorstore():
    """Load vector store from Supabase"""
    global db, is_loading

    try:
        logger.info("📥 Starting vector store download...")

        # Create local directories
        os.makedirs(LOCAL_PATH, exist_ok=True)
        os.makedirs("/app/.cache/huggingface", exist_ok=True)
        
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
            cache_folder="/app/.cache/huggingface"
        )

        logger.info("📚 Loading FAISS index...")
        db = FAISS.load_local(
            LOCAL_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )

        # Test the vector store
        test_results = db.similarity_search("test", k=1)
        logger.info(f"✅ Vector store loaded! Test search returned {len(test_results)} results")

        if test_results:
            logger.info(f"📄 Sample content: {test_results[0].page_content[:200]}...")

        is_loading = False
        logger.info("🎉 Vector store ready!")

    except Exception as e:
        logger.error(f"❌ Vector store loading failed: {str(e)}")
        logger.error(traceback.format_exc())
        is_loading = False
        db = None


def start_loading_vectorstore():
    """Start loading vector store in background thread"""
    thread = threading.Thread(target=load_vectorstore, daemon=True)
    thread.start()
    logger.info("🔄 Vector store loading in background...")


def get_answer(question, session_id=None, db_session=None):
    """Get answer using RAG system"""
    global db, gemini_client

    # Check if vector store is loaded
    if db is None:
        logger.warning("Vector store not loaded yet")
        return "The knowledge base is still loading. Please try again in a moment."

    try:
        search_query = question

        # Add context from chat history
        if session_id and db_session:
            chat_history = get_recent_messages(db_session, session_id)

            if chat_history:
                search_query = rewrite_question(chat_history, question)
                logger.info(f"🔁 Rewritten query: {search_query}")

        # Search for relevant documents
        docs = db.similarity_search(search_query, k=4)

        if not docs:
            logger.warning("⚠️ No relevant documents found")
            return (
                "I couldn't find relevant information in the Primis Digital knowledge base. "
                "Could you rephrase your question?"
            )

        logger.info(f"📚 Found {len(docs)} relevant documents")
        
        # Prepare context
        context = "\n\n---\n\n".join([doc.page_content for doc in docs])

        # Generate answer using Gemini
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

        if gemini_client:
            logger.info("🤖 Generating answer with Gemini...")
            response = gemini_client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt
            )
            answer = response.text
        else:
            logger.warning("Gemini client not available, using fallback")
            answer = "I found relevant information but the AI service is currently unavailable. Here are some key points from our knowledge base:\n\n" + context[:500] + "..."

        logger.info(f"✅ Answer generated: {len(answer)} characters")
        return answer

    except Exception as e:
        logger.error(f"❌ Error in get_answer: {str(e)}")
        logger.error(traceback.format_exc())
        return f"I encountered an error while processing your question. Please try again."


def get_recent_messages(db, session_id, limit=5):
    """Fetch recent chat messages for context"""
    try:
        chats = (
            db.query(Chat)
            .filter(Chat.session_id == session_id)
            .order_by(Chat.created_at.desc())
            .limit(limit)
            .all()
        )
        return list(reversed(chats))
    except Exception as e:
        logger.error(f"Error getting recent messages: {e}")
        return []


def rewrite_question(chat_history, user_question):
    """Convert follow-up questions into standalone questions"""
    if not gemini_client:
        return user_question
        
    try:
        conversation = ""
        for chat in chat_history:
            conversation += f"User: {chat.question}\nAssistant: {chat.answer}\n"

        prompt = f"""
You are a query rewriter.

Given the conversation below and a follow-up question,
rewrite the question so it can be understood independently.

Conversation:
{conversation}

Follow-up question:
{user_question}

Rewrite the question clearly:
"""

        response = gemini_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )

        return response.text.strip()
    except Exception as e:
        logger.error(f"Error rewriting question: {e}")
        return user_question


def is_vectorstore_ready():
    """Check if vector store is loaded and ready"""
    return db is not None and not is_loading
