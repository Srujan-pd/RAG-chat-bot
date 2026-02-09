import sys
import os
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

# Configure logging FIRST
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

logger.info("=" * 60)
logger.info("🚀 Starting Support AI Bot")
logger.info("=" * 60)

# Check critical environment variables
required_env_vars = {
    "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY"),
    "DATABASE_URL": os.getenv("DATABASE_URL"),
    "PORT": os.getenv("PORT", "8080")
}

for var_name, var_value in required_env_vars.items():
    if var_value:
        logger.info(f"✅ {var_name} is set")
    else:
        logger.error(f"❌ {var_name} is NOT set!")

# Import routers with error handling
try:
    from chat import router as chat_router
    logger.info("✅ Chat router imported")
except Exception as e:
    logger.error(f"❌ Failed to import chat router: {e}")
    chat_router = None

try:
    from voice_chat import router as voice_router
    logger.info("✅ Voice router imported")
except Exception as e:
    logger.error(f"❌ Failed to import voice router: {e}")
    voice_router = None

# Create FastAPI app
app = FastAPI(
    title="Support AI Bot",
    description="GenAI Support Chatbot with voice capabilities",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
logger.info("✅ CORS middleware configured")

# Database initialization on startup
@app.on_event("startup")
async def startup_event():
    logger.info("🔧 Running startup tasks...")
    try:
        from database import engine, Base
        import models
        
        # Create all tables
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Database tables synchronized")
        
        # Test database connection
        from sqlalchemy import text
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            logger.info("✅ Database connection verified")
            
    except Exception as e:
        logger.error(f"⚠️ Startup initialization error: {e}")
        logger.error(f"App will continue but may have limited functionality")

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
    """Root endpoint"""
    try:
        return RedirectResponse(url="/static/index.html")
    except:
        return {
            "status": "ok",
            "message": "Support AI Bot API",
            "docs": "/docs"
        }

# Include API routers with safety checks
if chat_router:
    app.include_router(chat_router, tags=["Chat"])
    logger.info("✅ Chat router included")
else:
    logger.warning("⚠️ Chat router not available")

if voice_router:
    app.include_router(voice_router, tags=["Voice"])
    logger.info("✅ Voice router included")
else:
    logger.warning("⚠️ Voice router not available")

# Mount static files directory (optional)
try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
    logger.info("✅ Static files mounted at /static")
except Exception as e:
    logger.warning(f"⚠️ Static files directory not found (this is OK): {e}")

logger.info("=" * 60)
logger.info("✅ Application initialized successfully")
logger.info("=" * 60)

# For local development
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    logger.info(f"🚀 Starting uvicorn on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
