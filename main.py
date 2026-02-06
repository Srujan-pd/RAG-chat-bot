from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
import logging
import os
import time
from chat import router as chat_router
from voice_chat import router as voice_router
from fastapi.staticfiles import StaticFiles
from rag_engine import start_loading_vectorstore, initialize_gemini, wait_for_vectorstore

app = FastAPI(title="Support AI Bot")

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

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check for Cloud Run"""
    from rag_engine import db, gemini_client
    from database import engine
    
    status = {
        "status": "healthy",
        "service": "Primis Digital AI Bot",
        "timestamp": time.time(),
        "services": {
            "database": engine is not None,
            "vectorstore": db is not None,
            "gemini": gemini_client is not None,
        }
    }
    return status

@app.get("/ready")
async def ready_check():
    """Readiness check - only returns 200 when fully ready"""
    from rag_engine import db, gemini_client
    from database import engine
    
    if engine is None or db is None or gemini_client is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    return {"status": "ready", "message": "All services initialized"}

@app.on_event("startup")
async def startup_event():
    """Initialize all services on startup - BLOCKS until ready"""
    logger.info("🚀 Starting Primis Digital AI Bot...")
    
    try:
        # Initialize database
        from database import engine, Base
        import models
        
        # Create tables if they don't exist
        if engine is not None:
            Base.metadata.create_all(bind=engine)
            logger.info("✅ Database tables synchronized")
        else:
            logger.warning("⚠️ Database engine not available")
        
        # Initialize AI services - BLOCKING
        logger.info("🤖 Initializing Gemini...")
        gemini_ready = initialize_gemini()
        
        if gemini_ready:
            logger.info("✅ Gemini initialized")
        else:
            logger.error("❌ Gemini initialization failed")
        
        logger.info("📚 Loading vector store...")
        
        # Start loading in background
        start_loading_vectorstore()
        
        # Wait for vector store with timeout
        logger.info("⏳ Waiting for vector store to load (max 60 seconds)...")
        vectorstore_ready = wait_for_vectorstore(timeout=60)
        
        if vectorstore_ready:
            logger.info("✅ Vector store loaded successfully")
        else:
            logger.warning("⚠️ Vector store not loaded, continuing without it")
        
        logger.info("🎉 All services initialized successfully!")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        import traceback
        traceback.print_exc()
        # Don't crash - continue with degraded service

# Include routers
app.include_router(chat_router)
app.include_router(voice_router)

# Root redirect
@app.get("/")
def root():
    return RedirectResponse(url="/static/index.html")

# Debug endpoint
@app.get("/debug")
async def debug_info():
    from database import engine
    from rag_engine import db, gemini_client
    
    return {
        "service": "Primis Digital AI Bot",
        "timestamp": time.time(),
        "services": {
            "database": "connected" if engine else "disconnected",
            "vectorstore": "loaded" if db else "loading/error",
            "gemini": "ready" if gemini_client else "not ready",
            "vectorstore_size": len(db.index.ntotal) if db else 0
        },
        "environment": {
            "port": os.getenv("PORT", "8080"),
            "node": os.getenv("K_REVISION", "unknown"),
        }
    }

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)
