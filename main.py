import logging
import os
import time
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from chat import router as chat_router
from voice_chat import router as voice_router
from rag_engine import start_loading_vectorstore, initialize_gemini, wait_for_vectorstore
from database import engine, Base
import models

# App Initialization
app = FastAPI(
    title="Primis Digital Support AI Bot",
    description="RAG-based AI chatbot with voice support",
    version="2.1.0"
)

# Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# CORS Configuration - More restrictive for production
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Startup Events
@app.on_event("startup")
async def startup_tasks():
    """Initialize all required services on startup"""
    logger.info("🚀 Starting Primis Digital Support AI Bot v2.1.0...")
    
    # Log environment info
    logger.info(f"Python version: {os.sys.version}")
    logger.info(f"PORT: {os.getenv('PORT', '8080')}")
    logger.info(f"CORS origins: {CORS_ORIGINS}")
    
    # Initialize Gemini
    logger.info("🤖 Initializing Gemini AI...")
    gemini_initialized = initialize_gemini()
    if not gemini_initialized:
        logger.error("❌ Failed to initialize Gemini - voice features may not work")
    
    # Initialize RAG system
    logger.info("📚 Starting RAG system...")
    start_loading_vectorstore()
    
    # Initialize Database
    try:
        if engine is not None:
            Base.metadata.create_all(bind=engine)
            logger.info("✅ Database tables synchronized")
        else:
            logger.warning("⚠️ Database engine not initialized - running without persistence")
    except Exception as e:
        logger.error(f"⚠️ DB initialization failed: {e}")
    
    # Wait a bit for vector store to start loading
    logger.info("⏳ Allowing vector store to initialize...")
    
    logger.info("✅ Startup complete!")

@app.on_event("shutdown")
async def shutdown_tasks():
    """Cleanup on shutdown"""
    logger.info("🛑 Shutting down...")

# Routers
app.include_router(chat_router)
app.include_router(voice_router)

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint for Cloud Run"""
    from rag_engine import db, is_loading, loading_error
    
    vectorstore_status = "loading" if is_loading else "ready" if db else "failed"
    
    return {
        "status": "healthy",
        "service": "Primis Digital Support AI Bot",
        "version": "2.1.0",
        "timestamp": time.time(),
        "components": {
            "vectorstore": vectorstore_status,
            "database": "connected" if engine else "disabled",
            "gemini": "initialized" if hasattr(initialize_gemini, '__call__') else "checking"
        }
    }

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Primis Digital Support AI Bot API",
        "version": "2.1.0",
        "endpoints": {
            "chat": "/chat/",
            "chat_history": "/chat/history/{user_id}",
            "voice_chat": "/voice/",
            "voice_tts": "/voice/tts",
            "health": "/health",
            "docs": "/docs"
        },
        "status": "running"
    }

# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc) if os.getenv("DEBUG") else "An error occurred. Please try again.",
            "contact": "contact@primisdigital.com"
        }
    )
