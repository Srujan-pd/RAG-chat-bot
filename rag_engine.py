import os
import traceback
import logging
from supabase_manager import SupabaseStorageManager
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from google import genai

# --------------------------------------------------
# Logging
# --------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --------------------------------------------------
# Config
# --------------------------------------------------
BUCKET_NAME = os.getenv("SUPABASE_BUCKET_NAME", "vectorstore-bucket")
REMOTE_FOLDER = "vectorstore"
LOCAL_PATH = "/tmp/vectorstore"

# --------------------------------------------------
# Globals (loaded ONCE at startup)
# --------------------------------------------------
vectorstore = None
gemini_client = None
init_error = None


# --------------------------------------------------
# INIT FUNCTIONS (CALLED AT STARTUP)
# --------------------------------------------------
def init_gemini():
    global gemini_client

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set")

    gemini_client = genai.Client(api_key=api_key)
    logger.info("✅ Gemini client initialized")


def init_vectorstore():
    global vectorstore

    logger.info("📥 Initializing vectorstore")

    os.makedirs(LOCAL_PATH, exist_ok=True)

    storage = SupabaseStorageManager()

    for file in ["index.faiss", "index.pkl"]:
        remote = f"{REMOTE_FOLDER}/{file}"
        local = os.path.join(LOCAL_PATH, file)

        logger.info(f"⬇️ Downloading {remote}")
        ok = storage.download_file(remote, local, BUCKET_NAME)

        if not ok or not os.path.exists(local) or os.path.getsize(local) == 0:
            raise RuntimeError(f"Failed to download {file}")

        logger.info(f"✅ {file} downloaded ({os.path.getsize(local):,} bytes)")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.load_local(
        LOCAL_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )

    logger.info("✅ Vectorstore loaded successfully")


def init_rag():
    """
    Called ONCE from FastAPI startup event
    """
    global init_error

    try:
        logger.info("🚀 RAG initialization started")

        init_gemini()
        init_vectorstore()

        # Sanity check
        test = vectorstore.similarity_search("test", k=1)
        logger.info(f"🧪 Vectorstore test OK ({len(test)} docs)")

        logger.info("🎉 RAG initialization completed")

    except Exception as e:
        init_error = str(e)
        logger.error("❌ RAG initialization failed")
        logger.error(traceback.format_exc())
        raise  # 🔥 CRASH CONTAINER (Cloud Run will restart)


# --------------------------------------------------
# RUNTIME FUNCTION (FAST)
# --------------------------------------------------
def get_answer(question: str, k: int = 4) -> str:
    if init_error:
        return f"❌ System initialization failed: {init_error}"

    if vectorstore is None or gemini_client is None:
        return "❌ System not ready. Please try again later."

    docs = vectorstore.similarity_search(question, k=k)

    if not docs:
        return (
            "I couldn't find relevant information in the Primis Digital "
            "knowledge base. Could you rephrase your question?"
        )

    context = "\n\n---\n\n".join(doc.page_content for doc in docs)

    prompt = f"""
You are a helpful assistant for Primis Digital.

Answer the user's question using ONLY the context below.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

    response = gemini_client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
    )

    return response.text

