import os
import threading
import traceback
import logging
import re
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
            raise ValueError("GEMINI_API_KEY not found in environment")

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

        test_results = db.similarity_search("test query", k=1)
        logger.info(f"✅ Vector store loaded! Test search returned {len(test_results)} results")

        if test_results:
            logger.info(f"📄 Sample content: {test_results[0].page_content[:200]}...")

        is_loading = False
        logger.info("🎉 Vector store ready!")

    except Exception as e:
        logger.error(f"❌ Vector store loading failed: {str(e)}")
        logger.error(traceback.format_exc())
        is_loading = False


def start_loading_vectorstore():
    """Start loading vector store in background thread"""
    thread = threading.Thread(target=load_vectorstore, daemon=True)
    thread.start()
    logger.info("🔄 Vector store loading in background...")


def is_greeting(question):
    """Check if the question is a greeting"""
    greetings = [
        'hello', 'hi', 'hey', 'good morning', 'good afternoon', 
        'good evening', 'greetings', 'howdy', 'hola', 'namaste',
        'hi there', 'hello there'
    ]
    question_lower = question.lower().strip()
    
    # Check for exact matches or greeting at the start of the message
    for greeting in greetings:
        if question_lower == greeting or question_lower.startswith(greeting + ' ') or question_lower.startswith(greeting + ','):
            return True
    
    return False


def is_network_error(error):
    """Check if the error is network-related"""
    network_keywords = [
        'connection', 'network', 'timeout', 'unreachable', 
        'dns', 'socket', 'http', 'ssl', 'certificate',
        'connection refused', 'connection reset', 'connection aborted',
        'failed to establish', 'name resolution', 'timed out'
    ]
    error_str = str(error).lower()
    return any(keyword in error_str for keyword in network_keywords)


def extract_links(docs):
    """Extract URLs from documents"""
    links = []
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    for doc in docs:
        found_urls = re.findall(url_pattern, doc.page_content)
        links.extend(found_urls)
    
    return list(set(links))  # Remove duplicates


def detect_query_type(question):
    """Detect if the query is about jobs, blogs, or services"""
    question_lower = question.lower()
    
    query_types = {
        'job': ['job', 'career', 'hiring', 'vacancy', 'position', 'employment', 'work with', 'join', 'opening', 'recruit'],
        'blog': ['blog', 'article', 'post', 'read', 'content', 'news', 'update'],
        'service': ['service', 'offering', 'solution', 'provide', 'product', 'technology', 'what do you do', 'what does']
    }
    
    detected_types = []
    for query_type, keywords in query_types.items():
        if any(keyword in question_lower for keyword in keywords):
            detected_types.append(query_type)
    
    return detected_types


def is_list_query(question):
    """Detect if user is asking for a complete list"""
    list_indicators = [
        'list', 'all', 'what are', 'show me', 'tell me about all',
        'services', 'offerings', 'products', 'solutions',
        'everything', 'complete', 'full list', 'entire'
    ]
    question_lower = question.lower()
    return any(indicator in question_lower for indicator in list_indicators)


def get_comprehensive_docs(db, query, k=10):
    """
    Get comprehensive document coverage by using multiple related searches
    This helps ensure we get ALL relevant content, not just top matches
    """
    all_docs = []
    seen_content = set()
    
    # Primary search
    primary_docs = db.similarity_search(query, k=k)
    
    for doc in primary_docs:
        content_hash = hash(doc.page_content[:200])  # Use first 200 chars as identifier
        if content_hash not in seen_content:
            all_docs.append(doc)
            seen_content.add(content_hash)
    
    # For service-related queries, do additional targeted searches
    query_lower = query.lower()
    additional_searches = []
    
    if any(word in query_lower for word in ['service', 'offering', 'solution', 'product']):
        additional_searches = [
            "services and solutions",
            "what we offer",
            "our offerings",
            "products and services"
        ]
    elif any(word in query_lower for word in ['job', 'career', 'hiring']):
        additional_searches = [
            "careers and jobs",
            "work with us",
            "join our team"
        ]
    elif any(word in query_lower for word in ['blog', 'article', 'post']):
        additional_searches = [
            "blog posts",
            "articles and content",
            "latest updates"
        ]
    
    # Perform additional searches
    for additional_query in additional_searches:
        additional_docs = db.similarity_search(additional_query, k=5)
        for doc in additional_docs:
            content_hash = hash(doc.page_content[:200])
            if content_hash not in seen_content:
                all_docs.append(doc)
                seen_content.add(content_hash)
    
    logger.info(f"📚 Comprehensive search returned {len(all_docs)} unique documents")
    return all_docs


def get_answer(question, session_id=None, db_session=None):
    """
    Get answer using RAG with conversational context.
    Enhanced version with greeting detection, link extraction, and comprehensive responses.
    """
    global db, gemini_client
    
    try:
        # Check if greeting
        if is_greeting(question):
            return (
                "Hello! 👋 I'm the Primis Digital support assistant. "
                "I can help you with information about our services, careers, blog posts, and more. "
                "How can I assist you today?"
            )
        
        # Check if vector store is ready
        if is_loading:
            return "The knowledge base is still loading. Please try again in a moment."
        
        if db is None:
            return (
                "I'm having trouble accessing the knowledge base right now. "
                "Please try again in a moment or contact our team directly."
            )
        
        if gemini_client is None:
            return "AI service is not available. Please contact support."
        
        # Get chat history for context if session provided
        search_query = question
        if session_id and db_session:
            try:
                chat_history = get_recent_messages(db_session, session_id, limit=5)
                if chat_history:
                    logger.info(f"🔄 Found {len(chat_history)} previous messages")
                    search_query = rewrite_question(chat_history, question)
                    logger.info(f"🔄 Rewritten query: {search_query}")
            except Exception as e:
                logger.error(f"⚠️ Error getting chat history: {str(e)}")
                # Continue with original question
        
        # Detect if this is a list query - if so, retrieve more documents
        is_asking_for_list = is_list_query(question)
        
        logger.info(f"📊 Query analysis - List query: {is_asking_for_list}")

        # Search for relevant documents - use comprehensive search for lists
        if is_asking_for_list:
            docs = get_comprehensive_docs(db, search_query, k=15)
            logger.info(f"📚 Using comprehensive search - Retrieved {len(docs)} documents")
        else:
            docs = db.similarity_search(search_query, k=6)
            logger.info(f"📚 Using standard search - Retrieved {len(docs)} documents")

        if not docs:
            logger.warning("⚠️ No relevant documents found")
            return (
                "I couldn't find specific information about that in our knowledge base. "
                "Please contact our team for further information. You can reach us through "
                "our website's contact form or email us directly."
            )

        # Log document details
        for i, doc in enumerate(docs):
            logger.info(f"  Doc {i+1}: {doc.page_content[:100]}...")

        # Extract links from documents
        links = extract_links(docs)
        if links:
            logger.info(f"🔗 Found {len(links)} links in documents: {links}")

        # Detect query type
        query_types = detect_query_type(question)
        if query_types:
            logger.info(f"🔍 Query types detected: {query_types}")

        context = "\n\n---\n\n".join([doc.page_content for doc in docs])

        # Build enhanced prompt based on query type
        if is_asking_for_list:
            specific_instructions = """
- **CRITICAL**: The user is asking for a LIST. You MUST provide a COMPLETE and COMPREHENSIVE list of ALL items found in the context
- Do NOT summarize or give examples - LIST EVERY SINGLE ITEM mentioned in the context
- Use bullet points or numbered lists for clarity
- Include brief descriptions for each item
- If services/products are mentioned, list ALL of them with their details
- Do not say "such as" or "including" - be exhaustive and complete"""
        else:
            specific_instructions = """
- Provide detailed and specific information
- If multiple items are mentioned, cover all of them
- Be thorough and complete in your response"""

        # Build enhanced prompt with link instructions
        prompt = f"""You are a helpful assistant for Primis Digital, a technology company.

Based on the following information from Primis Digital's website, answer the user's question accurately and professionally.

CONTEXT FROM PRIMIS DIGITAL:
{context}

USER QUESTION: {question}

GENERAL INSTRUCTIONS:
- Answer based ONLY on the provided context
- Be specific and cite relevant details
- If the context doesn't contain enough information to fully answer, end your response with: "Please contact our team for further information."
- Keep your answer professional and well-formatted
- Use clear paragraphs and formatting
{specific_instructions}

LINK HANDLING INSTRUCTIONS:
- **CRITICAL**: If there are any URLs/links in the context that are relevant to the question, you MUST include them in your answer
- For job-related queries: Include ALL career page links and mention how to apply
- For blog-related queries: Include ALL direct links to blog posts or articles  
- For service-related queries: Include ALL service page links with descriptions
- Format links clearly: either as clickable text or on separate lines
- If multiple relevant links exist, include ALL of them - do not omit any

ANSWER:"""

        logger.info("🤖 Generating answer with Gemini...")
        response = gemini_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )

        answer = response.text.strip()
        logger.info(f"✅ Answer generated: {len(answer)} characters")

        return answer

    except Exception as e:
        error_message = str(e)
        logger.error(f"❌ Error in get_answer: {error_message}")
        logger.error(traceback.format_exc())
        
        # Check if it's a network error
        if is_network_error(error_message):
            return (
                "We're experiencing network connectivity issues at the moment. "
                "Please try again in a few moments. If the problem persists, "
                "please contact our support team."
            )
        
        # Generic error response
        return (
            "I apologize, but I encountered an issue while processing your request. "
            "Please contact our team for further assistance."
        )


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
        logger.error(f"❌ Error fetching recent messages: {str(e)}")
        return []


def rewrite_question(chat_history, user_question):
    """Convert follow-up questions into standalone questions"""
    try:
        conversation = ""
        for chat in chat_history:
            conversation += f"User: {chat.question}\nAssistant: {chat.answer}\n"

        prompt = f"""You are a query rewriter.

Given the conversation below and a follow-up question, rewrite the question so it can be understood independently without the conversation context.

Conversation:
{conversation}

Follow-up question:
{user_question}

Rewritten standalone question:"""

        logger.info("🤖 Generating answer with Gemini...")
        response = gemini_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )

        rewritten = response.text.strip()
        return rewritten if rewritten else user_question
        
    except Exception as e:
        logger.error(f"❌ Error rewriting question: {str(e)}")
        return user_question  # Return original question if rewriting fails

