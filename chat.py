from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Chat Bot")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health endpoint (REQUIRED for Cloud Run)
@app.get("/health")
async def health():
    return {"status": "healthy", "service": "AI Chat Bot"}

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "AI Chat Bot API",
        "endpoints": {
            "chat": "POST /chat",
            "voice": "POST /voice",
            "health": "GET /health",
            "docs": "GET /docs"
        }
    }

# Import chat router
try:
    from chat import router as chat_router
    app.include_router(chat_router)
    logger.info("✅ Chat router loaded")
except Exception as e:
    logger.error(f"❌ Failed to load chat router: {e}")

# Import voice router
try:
    from voice_chat import router as voice_router
    app.include_router(voice_router)
    logger.info("✅ Voice router loaded")
except Exception as e:
    logger.error(f"❌ Failed to load voice router: {e}")

# Startup
@app.on_event("startup")
async def startup():
    logger.info("🚀 Starting AI Chat Bot...")
    
    # Initialize database
    try:
        from database import engine, Base
        import models
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Database initialized")
    except Exception as e:
        logger.error(f"⚠️ Database init error: {e}")
    
    # Initialize RAG system
    try:
        from rag_engine import initialize_gemini, start_loading_vectorstore
        initialize_gemini()
        start_loading_vectorstore()
        logger.info("✅ RAG system starting")
    except Exception as e:
        logger.error(f"⚠️ RAG init error: {e}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
