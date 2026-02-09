import os
import threading
import traceback
import logging
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
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
embedding_model = None
faiss_index = None
chunks = None
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
    """Load vector store from Supabase - simplified without langchain"""
    global embedding_model, faiss_index, chunks, is_loading

    try:
        logger.info("📥 Starting vector store download...")

        # Create local directories
        os.makedirs(LOCAL_PATH, exist_ok=True)
        os.makedirs("/app/.cache/huggingface", exist_ok=True)
        
        # Import here to avoid circular imports
        from supabase_manager import SupabaseStorageManager
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

        logger.info("🔧 Loading embedding model...")
        # Use sentence-transformers directly instead of langchain
        embedding_model = SentenceTransformer(
            'sentence-transformers/all-MiniLM-L6-v2',
            cache_folder="/app/.cache/huggingface"
        )

        logger.info("📚 Loading FAISS index...")
        # Load FAISS directly
        faiss_index = faiss.read_index(os.path.join(LOCAL_PATH, "index.faiss"))
        
        # Load chunks from pickle
        with open(os.path.join(LOCAL_PATH, "index.pkl"), 'rb') as f:
            data = pickle.load(f)
            # Handle different pickle formats
            if isinstance(data, dict) and 'texts' in data:
                chunks = data['texts']
            elif isinstance(data, list):
                chunks = data
            else:
                # Try to extract texts from langchain format
                chunks = [doc.page_content for doc in data] if hasattr(data[0], 'page_content') else []

        # Test the vector store
        if chunks:
            test_embedding = embedding_model.encode(["test query"])
            distances, indices = faiss_index.search(test_embedding, k=1)
            logger.info(f"✅ Vector store loaded! {len(chunks)} chunks, {faiss_index.ntotal} vectors")
            
            if indices[0][0] >= 0:
                sample = chunks[indices[0][0]][:200]
                logger.info(f"📄 Sample content: {sample}...")
        else:
            logger.warning("⚠️ No chunks loaded from vector store")

        is_loading = False
        logger.info("🎉 Vector store ready!")

    except Exception as e:
        logger.error(f"❌ Vector store loading failed: {str(e)}")
        logger.error(traceback.format_exc())
        is_loading = False
        embedding_model = None
        faiss_index = None
        chunks = None


def start_loading_vectorstore():
    """Start loading vector store in background thread"""
    thread = threading.Thread(target=load_vectorstore, daemon=True)
    thread.start()
    logger.info("🔄 Vector store loading in background...")


def get_answer(question, session_id=None, db_session=None):
    """Get answer using RAG system"""
    global embedding_model, faiss_index, chunks, gemini_client

    # Check if vector store is loaded
    if faiss_index is None or embedding_model is None:
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

        # Search for relevant documents using FAISS directly
        query_embedding = embedding_model.encode([search_query])
        distances, indices = faiss_index.search(query_embedding, k=4)
        
        # Get relevant chunks
        relevant_chunks = []
        for idx in indices[0]:
            if idx >= 0 and idx < len(chunks):
                relevant_chunks.append(chunks[idx])

        if not relevant_chunks:
            logger.warning("⚠️ No relevant documents found")
            return (
                "I couldn't find relevant information in the Primis Digital knowledge base. "
                "Could you rephrase your question?"
            )

        logger.info(f"📚 Found {len(relevant_chunks)} relevant documents")
        
        # Prepare context
        context = "\n\n---\n\n".join(relevant_chunks)

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
            answer = f"I found relevant information. Here are the key points:\n\n{relevant_chunks[0][:500]}..."

        logger.info(f"✅ Answer generated: {len(answer)} characters")
        return answer

    except Exception as e:
        logger.error(f"❌ Error in get_answer: {str(e)}")
        logger.error(traceback.format_exc())
        return "I encountered an error while processing your question. Please try again."


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
    if not gemini_client or not chat_history:
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
    return faiss_index is not None and not is_loading
