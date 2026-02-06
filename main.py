from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

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
    """Initialize database"""
    try:
        from database import engine, Base
        import models
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Database initialized")
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Primis Digital Support AI Bot API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "POST /chat/": "Chat with AI (form-data: text, user_id)",
            "GET /chat/history/{user_id}": "Get chat history",
            "POST /voice/": "Voice chat (form-data: file, user_id)",
            "GET /health": "Health check",
            "GET /debug": "Debug information"
        }
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    try:
        from rag_engine import db, loading_error, loading_complete, is_loading, load_attempts
        
        # Check database connection
        from database import SessionLocal
        db_session = SessionLocal()
        db_healthy = False
        try:
            db_session.execute("SELECT 1")
            db_healthy = True
        except:
            db_healthy = False
        finally:
            db_session.close()
        
        # Determine overall status
        if loading_error:
            status = "degraded"
        elif not db_healthy:
            status = "degraded"
        else:
            status = "healthy"
        
        vectorstore_status = "ready" if db is not None else ("error" if loading_complete else "loading")
        
        return {
            "status": status,
            "database": "healthy" if db_healthy else "unhealthy",
            "vectorstore_status": vectorstore_status,
            "vectorstore_ready": db is not None,
            "vectorstore_loading": is_loading,
            "load_attempts": load_attempts,
            "loading_error": loading_error if loading_error else None
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

@app.get("/debug")
async def debug():
    """Debug endpoint to check vector store status"""
    try:
        from rag_engine import db, loading_error, loading_complete, is_loading, load_attempts
        import os
        
        # Check local files
        vectorstore_path = "/tmp/vectorstore"
        local_files = []
        if os.path.exists(vectorstore_path):
            try:
                local_files = os.listdir(vectorstore_path)
                # Add file sizes
                local_files_with_size = []
                for f in local_files:
                    file_path = os.path.join(vectorstore_path, f)
                    if os.path.isfile(file_path):
                        size = os.path.getsize(file_path)
                        local_files_with_size.append(f"{f} ({size} bytes)")
                    else:
                        local_files_with_size.append(f"{f} (directory)")
                local_files = local_files_with_size
            except Exception as e:
                local_files = [f"error: {str(e)}"]
        
        return {
            "app": "running",
            "vector_db_loaded": db is not None,
            "vector_count": db.index.ntotal if db else 0,
            "loading_complete": loading_complete,
            "is_loading": is_loading,
            "load_attempts": load_attempts,
            "loading_error": loading_error,
            "environment_variables": {
                "supabase_bucket_set": bool(os.getenv("SUPABASE_BUCKET_NAME")),
                "supabase_url_set": bool(os.getenv("SUPABASE_URL")),
                "gemini_key_set": bool(os.getenv("GEMINI_API_KEY")),
                "database_url_set": bool(os.getenv("DATABASE_URL"))
            },
            "local_vectorstore_path": vectorstore_path,
            "local_files": local_files,
            "local_path_exists": os.path.exists(vectorstore_path),
            "port": os.getenv("PORT", "8080")
        }
    except Exception as e:
        return {
            "error": f"Debug check failed: {str(e)}",
            "traceback": str(e.__traceback__)
        }

# Import and include routers after app is created
from chat import router as chat_router
from voice_chat import router as voice_router

app.include_router(chat_router)
app.include_router(voice_router)

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
