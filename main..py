import os
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create app
app = FastAPI(
    title="AI Chat Bot",
    description="RAG-based AI Assistant for Primis Digital",
    version="1.0.0"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health Check Endpoint (REQUIRED for Cloud Run)
@app.get("/health")
async def health_check():
    """Health check endpoint for Cloud Run"""
    try:
        # Test database connection
        from database import engine
        with engine.connect() as conn:
            conn.execute("SELECT 1")
        db_status = "connected"
    except Exception as e:
        logger.warning(f"Database health check failed: {e}")
        db_status = "disconnected"
    
    return {
        "status": "healthy",
        "service": "AI Chat Bot",
        "database": db_status,
        "timestamp": os.getenv("DEPLOY_TIMESTAMP", "unknown")
    }

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "AI Chat Bot API",
        "service": "Primis Digital Support Assistant",
        "version": "1.0.0",
        "endpoints": {
            "chat": "POST /chat",
            "chat_history": "GET /chat/history/{user_id}",
            "health": "GET /health",
            "docs": "GET /docs"
        }
    }

# Include routers
try:
    from chat import router as chat_router
    app.include_router(chat_router)
    logger.info("✅ Chat router loaded successfully")
except Exception as e:
    logger.error(f"❌ Failed to load chat router: {e}")
    raise

# Conditionally include voice chat
try:
    from voice_chat import router as voice_router
    app.include_router(voice_router)
    logger.info("✅ Voice chat router loaded")
except ImportError:
    logger.info("ℹ️ Voice chat module not found - skipping")
except Exception as e:
    logger.warning(f"⚠️ Voice chat router failed: {e}")

# Startup event handler
@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    logger.info("🚀 Starting AI Chat Bot Application")
    
    # Setup cache directories
    cache_dirs = ["/tmp/huggingface", "/tmp/vectorstore"]
    for dir_path in cache_dirs:
        os.makedirs(dir_path, exist_ok=True)
    
    # Set environment variables
    os.environ["HF_HOME"] = "/tmp/huggingface"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # Initialize services
    try:
        # Initialize RAG system
        from rag_engine import initialize_gemini, start_loading_vectorstore
        
        gemini_ok = initialize_gemini()
        if gemini_ok:
            logger.info("✅ Gemini client initialized")
        else:
            logger.warning("⚠️ Gemini initialization failed")
        
        start_loading_vectorstore()
        logger.info("🔄 Vector store loading started in background")
        
    except Exception as e:
        logger.error(f"❌ RAG system initialization failed: {e}")
    
    # Initialize database
    try:
        from database import engine, Base
        import models
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Database tables synchronized")
    except Exception as e:
        logger.error(f"⚠️ Database initialization failed: {e}")
    
    logger.info("🎉 Application startup completed successfully")
