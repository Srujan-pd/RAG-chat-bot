from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

from chat import router as chat_router
from voice_chat import router as voice_router
from rag_engine import init_rag

app = FastAPI(
    title="Primis Digital Chatbot API",
    version="1.0.0"
)

# -------------------------------------------------
# Logging
# -------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------------------------
# CORS
# -------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------
# STARTUP (CRITICAL)
# -------------------------------------------------
@app.on_event("startup")
def startup_event():
    logger.info("🚀 Starting Primis Digital Chatbot...")

    # 1️⃣ Initialize DB
    try:
        from database import engine, Base
        import models  # noqa
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Database tables synchronized")
    except Exception as e:
        logger.error(f"❌ DB initialization failed: {e}")
        raise  # Fail fast (Cloud Run will restart)

    # 2️⃣ Initialize RAG (BLOCKING & SAFE)
    try:
        logger.info("🧠 Initializing RAG system...")
        init_rag()
        logger.info("🎉 RAG system ready")
    except Exception as e:
        logger.error(f"❌ RAG initialization failed: {e}")
        raise  # VERY IMPORTANT

# -------------------------------------------------
# Routers
# -------------------------------------------------
app.include_router(chat_router)
app.include_router(voice_router)

# -------------------------------------------------
# Root
# -------------------------------------------------
@app.get("/")
def root():
    return {
        "service": "Primis Digital Chatbot API",
        "status": "online",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "chat": "/chat/",
            "voice": "/voice/",
            "chat_history": "/chat/history/{user_id}",
        },
    }

# -------------------------------------------------
# Health (REAL readiness check)
# -------------------------------------------------
@app.get("/health")
def health():
    from rag_engine import vectorstore, gemini_client, init_error

    return {
        "status": "healthy" if vectorstore and gemini_client else "unhealthy",
        "vectorstore_loaded": vectorstore is not None,
        "gemini_initialized": gemini_client is not None,
        "init_error": init_error,
    }

