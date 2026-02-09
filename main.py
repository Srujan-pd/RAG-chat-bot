import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from chat import router as chat_router
from rag_engine import start_loading_vectorstore, initialize_gemini

# App Initialization
app = FastAPI(title="Support AI Bot")

# Logging Configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health Check Endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint for Cloud Run"""
    try:
        from database import engine
        with engine.connect() as conn:
            conn.execute("SELECT 1")
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "database": "disconnected", "error": str(e)}
        )

# Startup Events
@app.on_event("startup")
async def startup_tasks():
    """Initialize application on startup"""
    logger.info("🚀 Starting Support AI Bot...")
    
    # Set HuggingFace cache directory
    os.environ["HF_HOME"] = "/tmp/huggingface"
    os.makedirs("/tmp/huggingface", exist_ok=True)
    
    # Initialize Gemini
    logger.info("🔧 Initializing Gemini...")
    if not initialize_gemini():
        logger.warning("Gemini initialization failed - continuing without it")
    
    # Start vector store loading in background
    logger.info("🔄 Starting vector store loading...")
    start_loading_vectorstore()
    
    # Initialize Database
    try:
        from database import engine, Base
        import models  # Ensures models are registered
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Database tables synchronized")
    except Exception as e:
        logger.error(f"⚠️ Database initialization failed: {e}")

# Routers
app.include_router(chat_router)

# Conditionally include voice_chat if available
try:
    from voice_chat import router as voice_router
    app.include_router(voice_router)
    logger.info("✅ Voice chat router included")
except ImportError:
    logger.info("ℹ️ Voice chat router not found - skipping")
except Exception as e:
    logger.error(f"⚠️ Error loading voice chat: {e}")

# Root endpoint
@app.get("/")
def root():
    return {
        "message": "Support AI Bot API", 
        "status": "running", 
        "docs": "/docs",
        "endpoints": {
            "chat": "/chat",
            "health": "/health"
        }
    }
