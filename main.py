from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
import logging
import os

# Import routers
from chat import router as chat_router
from voice_chat import router as voice_router

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Support AI Bot",
    description="GenAI Support Chatbot with voice capabilities",
    version="1.0.0"
)

# CORS middleware (allows frontend to communicate with backend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change in production to specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database initialization on startup
@app.on_event("startup")
async def startup_event():
    try:
        from database import engine, Base
        import models
        
        # Create all tables
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Database tables synchronized")
        logger.info(f"✅ Gemini API Key present: {bool(os.getenv('GEMINI_API_KEY'))}")
        logger.info(f"✅ Database URL present: {bool(os.getenv('DATABASE_URL'))}")
    except Exception as e:
        logger.error(f"⚠️ Startup initialization failed: {e}")

# Health check endpoint (required for Cloud Run)
@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "support-ai-bot",
        "version": "1.0.0"
    }

# Root endpoint
@app.get("/")
def root():
    """Root endpoint - redirects to static page"""
    return RedirectResponse(url="/static/index.html")

# Include API routers
app.include_router(chat_router, tags=["Chat"])
app.include_router(voice_router, tags=["Voice"])

# Mount static files directory (if exists)
try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
    logger.info("✅ Static files mounted at /static")
except Exception as e:
    logger.warning(f"⚠️ Static files directory not found: {e}")

# For local development
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
