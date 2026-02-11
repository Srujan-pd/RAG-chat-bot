import logging
import os
import time
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from chat import router as chat_router
from voice_chat import router as voice_router
from rag_engine import start_loading_vectorstore, initialize_gemini, wait_for_vectorstore, db, is_loading
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
    
    # Initialize Database
    try:
        if engine is not None:
            Base.metadata.create_all(bind=engine)
            logger.info("✅ Database tables synchronized")
        else:
            logger.warning("⚠️ Database engine not initialized - running without persistence")
    except Exception as e:
        logger.error(f"⚠️ DB initialization failed: {e}")
    
    # CRITICAL FIX: Start RAG system and WAIT for it to complete
    logger.info("📚 Starting RAG system and waiting for vector store to load...")
    start_loading_vectorstore()
    
    # Wait for vector store with extended timeout for cold starts
    logger.info("⏳ Waiting up to 180 seconds for vector store to load...")
    if wait_for_vectorstore(timeout=180):
        logger.info("✅ Vector store loaded successfully during startup!")
        from rag_engine import db
        if db:
            # Test the vector store
            try:
                test_results = db.similarity_search("test", k=1)
                logger.info(f"✅ Vector store test successful - found {len(test_results)} results")
            except Exception as e:
                logger.error(f"⚠️ Vector store test failed: {e}")
    else:
        logger.error("❌ Vector store failed to load during startup - will retry on requests")
    
    logger.info("✅ Startup complete! Application is ready to handle requests.")

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
            "vectorstore": {
                "status": vectorstore_status,
                "loaded": db is not None,
                "loading": is_loading,
                "error": loading_error
            },
            "database": "connected" if engine else "disabled",
            "gemini": "initialized" if gemini_initialized else "failed"
        },
        "uptime": time.time() - startup_time if 'startup_time' in globals() else 0
    }

# Track startup time
startup_time = time.time()

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    from rag_engine import db, is_loading
    
    vectorstore_status = "loading" if is_loading else "ready" if db else "not_loaded"
    
    return {
        "message": "Primis Digital Support AI Bot API",
        "version": "2.1.0",
        "status": "running",
        "vectorstore": vectorstore_status,
        "endpoints": {
            "chat": "/chat/",
            "chat_history": "/chat/history/{user_id}",
            "voice_chat": "/voice/",
            "voice_tts": "/voice/tts",
            "health": "/health",
            "docs": "/docs"
        }
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

# Liveness probe endpoint for Cloud Run
@app.get("/health/liveness")
async def liveness_check():
    """Liveness probe for Cloud Run"""
    return {"status": "alive"}

# Readiness probe endpoint for Cloud Run
@app.get("/health/readiness")
async def readiness_check():
    """Readiness probe for Cloud Run - only returns 200 when vector store is ready"""
    from rag_engine import db, is_loading
    
    if db is not None:
        return {"status": "ready", "vectorstore": "loaded"}
    elif not is_loading:
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "vectorstore": "failed"}
        )
    else:
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "vectorstore": "loading"}
        )
