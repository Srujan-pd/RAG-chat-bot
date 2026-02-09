import os
import logging
import traceback
from supabase_manager import SupabaseStorageManager
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from google import genai

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BUCKET_NAME = os.getenv("SUPABASE_BUCKET_NAME", "vectorstore-bucket")
REMOTE_FOLDER = "vectorstore"
LOCAL_PATH = "/tmp/vectorstore"

vectorstore = None
gemini_client = None
init_error = None


def init_gemini():
    global gemini_client
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set")
    gemini_client = genai.Client(api_key=api_key)
    logger.info("✅ Gemini initialized")


def init_vectorstore():
    global vectorstore

    os.makedirs(LOCAL_PATH, exist_ok=True)
    storage = SupabaseStorageManager()

    for file in ["index.faiss", "index.pkl"]:
        remote = f"{REMOTE_FOLDER}/{file}"
        local = os.path.join(LOCAL_PATH, file)

        logger.info(f"⬇️ Downloading {remote}")
        ok = storage.download_file(remote, local, BUCKET_NAME)

        if not ok or not os.path.exists(local) or os.path.getsize(local) == 0:
            raise RuntimeError(f"Failed to download {file}")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.load_local(
        LOCAL_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )

    logger.info("✅ Vectorstore loaded")


def init_rag():
    global init_error
    try:
        logger.info("🚀 RAG initialization started")
        init_gemini()
        init_vectorstore()
        logger.info("🎉 RAG ready")
    except Exception as e:
        init_error = str(e)
        logger.error("❌ RAG initialization failed")
        logger.error(traceback.format_exc())
        raise


def get_answer(question: str, k: int = 4) -> str:
    if init_error:
        return f"❌ Initialization failed: {init_error}"

    if not vectorstore or not gemini_client:
        return "❌ System not ready"

    docs = vectorstore.similarity_search(question, k=k)
    if not docs:
        return "I couldn’t find relevant information. Please rephrase."

    context = "\n\n".join(d.page_content for d in docs)

    prompt = f"""
Use ONLY the context below to answer.

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

