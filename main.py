from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
from chat import router as chat_router
from voice_chat import router as voice_router
from rag_engine import load_vectorstore_async

app = FastAPI(title="Primis Digital Support Bot")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Initialize database and start vector store loading"""
    # Create database tables
    from database import engine, Base
    import models
    Base.metadata.create_all(bind=engine)
    logger.info("✅ Database initialized")
    
    # Load vector store from GCS in background (non-blocking)
    load_vectorstore_async()
    logger.info("🔄 Vector store loading started in background")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Support AI Bot API",
        "version": "0.1.0",
        "status": "running"
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    from rag_engine import db, loading_error, loading_complete
    
    status = "healthy"
    vectorstore_status = "ready" if db is not None else ("error" if loading_complete else "loading")
    
    if loading_error:
        status = "degraded"
    
    return {
        "status": status,
        "vectorstore_status": vectorstore_status,
        "vectorstore_ready": db is not None,
        "loading_complete": loading_complete,
        "error": loading_error if loading_error else None
    }

@app.get("/debug")
async def debug():
    """Debug endpoint to check vector store status"""
    from rag_engine import db, loading_error, loading_complete
    import os
    
    # Check local files
    vectorstore_path = "/tmp/vectorstore"
    local_files = []
    if os.path.exists(vectorstore_path):
        try:
            local_files = os.listdir(vectorstore_path)
        except:
            local_files = ["error reading directory"]
    
    return {
        "vector_db_loaded": db is not None,
        "loading_complete": loading_complete,
        "loading_error": loading_error,
        "gcs_bucket": os.getenv("GCS_BUCKET_NAME"),
        "gemini_key_exists": os.getenv("GEMINI_API_KEY") is not None,
        "local_vectorstore_path": vectorstore_path,
        "local_files": local_files,
        "local_path_exists": os.path.exists(vectorstore_path)
    }

# Include routers
app.include_router(chat_router)
app.include_router(voice_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
