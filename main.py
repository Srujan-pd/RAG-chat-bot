from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
import logging
import os
from chat import router as chat_router
from voice_chat import router as voice_router
from fastapi.staticfiles import StaticFiles
from rag_engine import start_loading_vectorstore, initialize_gemini

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

# Add health check endpoint (CRITICAL for Cloud Run)
@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "Primis Digital AI Bot is running"}

@app.on_event("startup")
async def startup_event():
    """Initialize all services on startup"""
    try:
        # Initialize database
        from database import engine, Base
        import models
        
        # Create tables if they don't exist
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Database tables synchronized")
        
        # Initialize AI services
        logger.info("🚀 Starting RAG system...")
        initialize_gemini()
        start_loading_vectorstore()
        
    except Exception as e:
        logger.error(f"⚠️ Startup initialization failed: {e}")
        # Don't crash the app, continue without some features
        import traceback
        traceback.print_exc()

# Include routers
app.include_router(chat_router)
app.include_router(voice_router)

# Root redirect
@app.get("/")
def root():
    return RedirectResponse(url="/static/index.html")

# Mount static files LAST (important for route priority)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Add a catch-all route for debugging
@app.get("/debug")
async def debug_info():
    from database import DATABASE_URL, engine
    from rag_engine import db, gemini_client
    
    return {
        "database_connected": engine is not None,
        "vectorstore_loaded": db is not None,
        "gemini_ready": gemini_client is not None,
        "port": os.getenv("PORT", "Not set"),
    }

if __name__ == "__main__":
    port = int(os.getenv("PORT", 9000))
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)
