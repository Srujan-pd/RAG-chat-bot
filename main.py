from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
import logging
import os
import time
from fastapi.staticfiles import StaticFiles

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
    return {
        "status": "healthy",
        "service": "Primis Digital AI Bot",
        "timestamp": time.time(),
        "version": "1.0.0"
    }

@app.get("/ready")
async def ready_check():
    """Readiness check - returns 200 when services are ready"""
    from rag_engine import db, gemini_client, is_loading, load_error
    
    if is_loading:
        raise HTTPException(status_code=503, detail=f"Vector store loading... Error: {load_error}")
    
    if db is None:
        raise HTTPException(status_code=503, detail=f"Vector store not loaded. Error: {load_error}")
    
    if gemini_client is None:
        raise HTTPException(status_code=503, detail="Gemini not initialized")
    
    return {
        "status": "ready",
        "services": {
            "vectorstore": "loaded",
            "gemini": "ready",
            "database": "ready"
        }
    }

@app.on_event("startup")
async def startup_event():
    """Initialize all services on startup"""
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
        
        # Initialize AI services
        try:
            from rag_engine import initialize_gemini, start_loading_vectorstore
            
            logger.info("🤖 Initializing Gemini...")
            gemini_ready = initialize_gemini()
            
            if gemini_ready:
                logger.info("✅ Gemini initialized")
            else:
                logger.error("❌ Gemini initialization failed - check GEMINI_API_KEY")
            
            logger.info("📚 Loading vector store in background...")
            start_loading_vectorstore()
            logger.info("🔄 Vector store loading started")
            
        except ImportError as e:
            logger.error(f"❌ Could not import rag_engine: {e}")
        except Exception as e:
            logger.error(f"❌ Error initializing AI services: {e}")
            import traceback
            traceback.print_exc()
        
        logger.info("🎉 Services initialized!")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        import traceback
        traceback.print_exc()

# Import and include routers after initialization
try:
    from chat import router as chat_router
    app.include_router(chat_router)
    logger.info("✅ Chat router loaded")
except ImportError as e:
    logger.error(f"❌ Failed to load chat router: {e}")

try:
    from voice_chat import router as voice_router
    app.include_router(voice_router)
    logger.info("✅ Voice router loaded")
except ImportError as e:
    logger.warning(f"⚠️ Failed to load voice router: {e}")

# Root redirect
@app.get("/")
def root():
    return RedirectResponse(url="/static/index.html")

# Debug endpoint
@app.get("/debug")
async def debug_info():
    """Debug endpoint to check service status"""
    try:
        from database import engine
        from rag_engine import db, gemini_client, is_loading, load_error
        
        # Check file existence
        faiss_exists = os.path.exists("/tmp/vectorstore/index.faiss") if os.path.exists("/tmp/vectorstore") else False
        pkl_exists = os.path.exists("/tmp/vectorstore/index.pkl") if os.path.exists("/tmp/vectorstore") else False
        
        return {
            "service": "Primis Digital AI Bot",
            "timestamp": time.time(),
            "services": {
                "database": "connected" if engine else "disconnected",
                "vectorstore_loaded": db is not None,
                "vectorstore_loading": is_loading,
                "vectorstore_error": load_error,
                "gemini_ready": gemini_client is not None,
                "files": {
                    "faiss_exists": faiss_exists,
                    "pkl_exists": pkl_exists
                }
            },
            "environment": {
                "supabase_url_set": bool(os.getenv("SUPABASE_URL")),
                "supabase_key_set": bool(os.getenv("SUPABASE_KEY")),
                "gemini_key_set": bool(os.getenv("GEMINI_API_KEY")),
                "database_url_set": bool(os.getenv("DATABASE_URL"))
            }
        }
    except Exception as e:
        return {"error": str(e), "timestamp": time.time()}

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)
