import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from chat import router as chat_router
from voice_chat import router as voice_router
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
    os.environ["HF_HOME"] = "/app/.cache/huggingface"
    os.makedirs("/app/.cache/huggingface", exist_ok=True)
    
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
app.include_router(voice_router)

# Root Redirect
@app.get("/")
def root():
    return RedirectResponse(url="/static/index.html")

# Static Files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Favicon handler
@app.get("/favicon.ico")
async def favicon():
    return RedirectResponse(url="/static/favicon.ico")
