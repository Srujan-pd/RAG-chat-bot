import os
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create app
app = FastAPI(title="AI Chat Bot")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health endpoint
@app.get("/health")
async def health():
    return {"status": "healthy"}

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "AI Chat Bot API",
        "endpoints": {
            "chat": "/chat",
            "health": "/health",
            "docs": "/docs"
        }
    }

# Import routers
try:
    from chat import router as chat_router
    app.include_router(chat_router)
    logger.info("✅ Chat router loaded")
except Exception as e:
    logger.error(f"❌ Failed to load chat router: {e}")

try:
    from voice_chat import router as voice_router
    app.include_router(voice_router)
    logger.info("✅ Voice chat router loaded")
except ImportError:
    logger.info("ℹ️ Voice chat not available")
except Exception as e:
    logger.error(f"⚠️ Voice chat error: {e}")

# Startup
@app.on_event("startup")
async def startup():
    logger.info("🚀 Starting AI Chat Bot")
    
    # Initialize RAG
    try:
        from rag_engine import start_loading_vectorstore, initialize_gemini
        initialize_gemini()
        start_loading_vectorstore()
        logger.info("✅ RAG system initialized")
    except Exception as e:
        logger.error(f"⚠️ RAG init failed: {e}")
    
    # Initialize database
    try:
        from database import engine, Base
        import models
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Database initialized")
    except Exception as e:
        logger.error(f"⚠️ Database init failed: {e}")
