from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
from chat import router as chat_router
from voice_chat import router as voice_router
from rag_engine import start_loading_vectorstore, initialize_gemini

app = FastAPI(title="Primis Digital Chatbot API")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Initialize database and RAG system"""
    logger.info("🚀 Starting Primis Digital Chatbot...")
    
    # Initialize database
    try:
        from database import engine, Base
        import models
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Database tables synchronized")
    except Exception as e:
        logger.error(f"⚠️  DB initialization failed: {e}")
    
    # Initialize RAG system
    logger.info("🔑 Initializing Gemini and RAG system...")
    initialize_gemini()
    start_loading_vectorstore()

# Include routers
app.include_router(chat_router)
app.include_router(voice_router)

# Root endpoint
@app.get("/")
def root():
    """API root endpoint"""
    return {
        "service": "Primis Digital Chatbot API",
        "status": "online",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "chat": "/chat/",
            "voice": "/voice/",
            "chat_history": "/chat/history/{user_id}"
        }
    }

# Health check endpoint
@app.get("/health")
def health():
    """Health check for Cloud Run"""
    from rag_engine import is_loading, gemini_client
    
    return {
        "status": "healthy",
        "vectorstore_status": "loaded" if not is_loading else "loading",
        "gemini_initialized": gemini_client is not None
    }

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
