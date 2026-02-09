from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

from chat import router as chat_router
from voice_chat import router as voice_router
from rag_engine import init_rag

app = FastAPI(title="Primis Digital Chatbot API")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup_event():
    logger.info("🚀 Starting Primis Digital Chatbot...")

    try:
        from database import engine, Base
        import models
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Database ready")
    except Exception:
        logger.exception("❌ Database init failed")
        raise

    try:
        logger.info("🧠 Initializing RAG...")
        init_rag()
        logger.info("🎉 RAG initialized")
    except Exception:
        logger.exception("❌ RAG init failed")
        raise


app.include_router(chat_router)
app.include_router(voice_router)

@app.get("/health")
def health():
    from rag_engine import vectorstore, gemini_client, init_error
    return {
        "status": "healthy" if vectorstore and gemini_client else "unhealthy",
        "vectorstore": vectorstore is not None,
        "gemini": gemini_client is not None,
        "error": init_error,
    }

