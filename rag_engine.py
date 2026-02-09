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
load_error = None


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
    global embedding_model, faiss_index, chunks, is_loading, load_error
    
    load_error = None
    
    try:
        logger.info("📥 Starting vector store download...")
        
        # Create local directory
        os.makedirs(LOCAL_PATH, exist_ok=True)
        
        # Try to download from Supabase
        try:
            from supabase_manager import SupabaseStorageManager
            storage = SupabaseStorageManager()
            
            files_to_download = ["index.faiss", "index.pkl"]
            for filename in files_to_download:
                remote_path = f"{REMOTE_FOLDER}/{filename}"
                local_file = os.path.join(LOCAL_PATH, filename)
                
                logger.info(f"⬇️  Downloading {remote_path}...")
                if storage.download_file(remote_path, local_file, BUCKET_NAME):
                    logger.info(f"✅ Downloaded {filename}")
                else:
                    raise Exception(f"Failed to download {filename}")
        except Exception as e:
            logger.warning(f"⚠️ Supabase download failed: {e}")
            # Check if files exist locally
            if not all(os.path.exists(os.path.join(LOCAL_PATH, f)) for f in ["index.faiss", "index.pkl"]):
                raise Exception("Vector store files not found")

        logger.info("🔧 Loading embedding model...")
        embedding_model = SentenceTransformer(
            'all-MiniLM-L6-v2',
            cache_folder="/tmp/huggingface"
        )

        logger.info("📚 Loading FAISS index...")
        faiss_index = faiss.read_index(os.path.join(LOCAL_PATH, "index.faiss"))
        
        # Load chunks from pickle
        with open(os.path.join(LOCAL_PATH, "index.pkl"), 'rb') as f:
            data = pickle.load(f)
            
            # Handle different data formats
            if isinstance(data, dict):
                if 'texts' in data:
                    chunks = data['texts']
                elif 'chunks' in data:
                    chunks = data['chunks']
                else:
                    chunks = list(data.values())[0] if data else []
            elif isinstance(data, list):
                chunks = data
            else:
                # Try to extract from langchain format
                if hasattr(data, '__iter__'):
                    chunks = [doc.page_content for doc in data if hasattr(doc, 'page_content')]
                else:
                    chunks = []

        if not chunks:
            raise Exception("No chunks loaded from vector store")
        
        logger.info(f"✅ Vector store loaded: {len(chunks)} chunks, {faiss_index.ntotal} vectors")
        is_loading = False
        
        # Test search
        test_query = "hello"
        test_embedding = embedding_model.encode([test_query])
        distances, indices = faiss_index.search(test_embedding, k=1)
        if indices[0][0] >= 0:
            logger.info("🧪 Vector store test: PASSED")
        else:
            logger.warning("🧪 Vector store test: No results found")
        
    except Exception as e:
        error_msg = f"Vector store loading failed: {str(e)}"
        logger.error(f"❌ {error_msg}")
        logger.error(traceback.format_exc())
        load_error = error_msg
        is_loading = False
        embedding_model = None
        faiss_index = None
        chunks = None


def start_loading_vectorstore():
    """Start loading vector store in background thread"""
    if is_loading:
        thread = threading.Thread(target=load_vectorstore, daemon=True)
        thread.start()
        logger.info("🔄 Vector store loading in background...")
    else:
        logger.info("ℹ️ Vector store already loaded or loading")


def get_answer(question, session_id=None, db_session=None):
    """Get answer using RAG system"""
    global embedding_model, faiss_index, chunks, gemini_client
    
    # Check if vector store is loaded
    if faiss_index is None or embedding_model is None:
        if load_error:
            return f"Knowledge base error: {load_error}"
        return "Knowledge base is loading. Please try again in a moment."

    try:
        search_query = question

        # Add context from chat history
        if session_id and db_session:
            try:
                chat_history = get_recent_messages(db_session, session_id)
                if chat_history:
                    search_query = rewrite_question(chat_history, question)
                    logger.info(f"🔁 Rewritten query: {search_query}")
            except Exception as e:
                logger.warning(f"Failed to rewrite question: {e}")

        # Search for relevant documents
        query_embedding = embedding_model.encode([search_query])
        distances, indices = faiss_index.search(query_embedding, k=3)
        
        # Get relevant chunks
        relevant_chunks = []
        for idx in indices[0]:
            if 0 <= idx < len(chunks):
                relevant_chunks.append(chunks[idx])
        
        if not relevant_chunks:
            return "I couldn't find relevant information in our knowledge base. Could you rephrase your question?"

        logger.info(f"📚 Found {len(relevant_chunks)} relevant documents")
        
        # Prepare context (limit to avoid token limits)
        context = "\n\n---\n\n".join(relevant_chunks[:3])
        if len(context) > 4000:  # Safety limit
            context = context[:4000] + "..."
        
        # Generate answer using Gemini
        prompt = f"""You are a helpful AI assistant for Primis Digital.

Based on this information from Primis Digital's website:

{context}

User Question: {question}

Instructions:
1. Answer based ONLY on the provided context
2. Be specific and professional
3. If context doesn't have enough information, say so politely
4. Keep answer concise (2-3 paragraphs max)

Answer:"""

        if gemini_client:
            logger.info("🤖 Generating answer with Gemini...")
            response = gemini_client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt
            )
            answer = response.text.strip()
        else:
            logger.warning("Gemini not available, using fallback")
            answer = f"Based on our information: {relevant_chunks[0][:300]}..."

        logger.info(f"✅ Answer generated ({len(answer)} chars)")
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
        for chat in chat_history[:3]:  # Limit to last 3 exchanges
            conversation += f"User: {chat.question}\nAssistant: {chat.answer}\n"

        prompt = f"""
Given this conversation:
{conversation}

And this follow-up question: {user_question}

Rewrite the question to be clear and standalone:
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


def get_vectorstore_status():
    """Get detailed vector store status"""
    return {
        "loaded": faiss_index is not None,
        "loading": is_loading,
        "chunks_count": len(chunks) if chunks else 0,
        "error": load_error,
        "gemini_ready": gemini_client is not None
    }
