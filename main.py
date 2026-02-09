from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
import datetime
from contextlib import asynccontextmanager

from chat import router as chat_router
from voice_chat import router as voice_router
from rag_engine import init_rag

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global state
app_state = {
    "database_ready": False,
    "rag_ready": False,
    "startup_time": None,
    "startup_errors": []
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events
    """
    # Startup
    logger.info("🚀 Starting Primis Digital Chatbot...")
    app_state["startup_time"] = datetime.datetime.utcnow()
    
    # Database initialization
    try:
        from database import init_database, Base
        import models  # noqa - to ensure models are registered
        
        # Initialize database with retry
        engine = init_database(max_retries=3, retry_delay=2)
        if engine:
            Base.metadata.create_all(bind=engine)
            app_state["database_ready"] = True
            logger.info("✅ Database initialized successfully")
        else:
            error_msg = "Database connection failed after retries"
            app_state["startup_errors"].append(error_msg)
            logger.error(f"❌ {error_msg}")
            
    except Exception as e:
        error_msg = f"Database initialization failed: {str(e)}"
        app_state["startup_errors"].append(error_msg)
        logger.exception("❌ Database initialization failed")
    
    # RAG initialization (non-blocking - starts in background)
    try:
        logger.info("🧠 Starting RAG initialization...")
        # This should be non-blocking or have its own timeout
        init_rag()
        app_state["rag_ready"] = True
        logger.info("🎉 RAG initialization started successfully")
    except Exception as e:
        error_msg = f"RAG initialization failed: {str(e)}"
        app_state["startup_errors"].append(error_msg)
        logger.exception("❌ RAG initialization failed")
    
    logger.info("✅ Application startup completed")
    yield
    
    # Shutdown (if needed)
    logger.info("👋 Shutting down application...")

app = FastAPI(
    title="Primis Digital Chatbot API",
    version="1.0.0",
    description="AI Chatbot for Primis Digital with RAG capabilities",
    lifespan=lifespan
)

# -----------------------------
# CORS
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# ROUTERS
# -----------------------------
app.include_router(chat_router)
app.include_router(voice_router)

# -----------------------------
# ROOT ENDPOINT
# -----------------------------
@app.get("/")
def root():
    """API root endpoint"""
    return {
        "service": "Primis Digital Chatbot API",
        "status": "online",
        "version": "1.0.0",
        "uptime": str(datetime.datetime.utcnow() - app_state["startup_time"]) if app_state["startup_time"] else "unknown",
        "endpoints": {
            "health": "/health",
            "chat": "/chat/",
            "voice": "/voice/",
            "chat_history": "/chat/history/{user_id}",
            "docs": "/docs",
            "redoc": "/redoc"
        }
    }

# -----------------------------
# HEALTH CHECK ENDPOINT
# -----------------------------
@app.get("/health")
def health():
    """Comprehensive health check for Cloud Run"""
    from rag_engine import vectorstore, gemini_client, init_error, is_rag_initialized
    
    health_status = {
        "status": "healthy",
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "uptime": str(datetime.datetime.utcnow() - app_state["startup_time"]) if app_state["startup_time"] else "unknown",
        "components": {}
    }
    
    # Check database
    try:
        from database import engine
        if engine is None:
            health_status["components"]["database"] = {
                "status": "disconnected",
                "message": "Database engine not initialized"
            }
            health_status["status"] = "degraded"
        else:
            with engine.connect() as conn:
                conn.execute("SELECT 1")
            health_status["components"]["database"] = {
                "status": "connected",
                "message": "OK"
            }
    except Exception as e:
        health_status["components"]["database"] = {
            "status": "error",
            "message": str(e)[:100]
        }
        health_status["status"] = "degraded"
    
    # Check RAG components
    # Vector store
    if vectorstore is not None:
        health_status["components"]["vectorstore"] = {
            "status": "loaded",
            "message": "OK"
        }
    elif init_error:
        health_status["components"]["vectorstore"] = {
            "status": "error",
            "message": init_error[:100] if init_error else "Unknown error"
        }
        health_status["status"] = "degraded"
    elif is_rag_initialized:
        health_status["components"]["vectorstore"] = {
            "status": "loading",
            "message": "Vector store is being loaded"
        }
        # Still considered healthy if loading in background
    else:
        health_status["components"]["vectorstore"] = {
            "status": "not_initialized",
            "message": "Vector store initialization not started"
        }
        health_status["status"] = "degraded"
    
    # Gemini client
    if gemini_client is not None:
        health_status["components"]["gemini"] = {
            "status": "initialized",
            "message": "OK"
        }
    else:
        health_status["components"]["gemini"] = {
            "status": "not_initialized",
            "message": "Gemini client not available"
        }
        health_status["status"] = "degraded"
    
    # Overall service readiness
    # The service is considered ready if database is connected
    # RAG components can be loading in background
    if health_status["components"].get("database", {}).get("status") != "connected":
        health_status["ready"] = False
    else:
        health_status["ready"] = True
    
    # Add startup errors if any
    if app_state["startup_errors"]:
        health_status["startup_errors"] = app_state["startup_errors"]
        health_status["status"] = "degraded"
    
    return health_status

# -----------------------------
# READINESS PROBE (for Kubernetes/Cloud Run)
# -----------------------------
@app.get("/ready")
def ready():
    """Readiness probe - checks if service is ready to accept traffic"""
    from rag_engine import is_rag_initialized
    
    # Basic readiness check
    # Service is ready if database is connected
    try:
        from database import engine
        if engine is None:
            return {"ready": False, "reason": "Database not initialized"}
        
        with engine.connect() as conn:
            conn.execute("SELECT 1")
        
        # Database is connected, service is ready
        # RAG can load in background
        return {"ready": True, "message": "Service is ready"}
        
    except Exception as e:
        return {"ready": False, "reason": f"Database error: {str(e)[:100]}"}

# -----------------------------
# LIVENESS PROBE
# -----------------------------
@app.get("/alive")
def alive():
    """Liveness probe - checks if service is running"""
    return {
        "alive": True,
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "service": "Primis Digital Chatbot API"
    }

# -----------------------------
# ERROR HANDLERS
# -----------------------------
@app.exception_handler(500)
async def internal_server_error_handler(request, exc):
    """Handle 500 errors gracefully"""
    logger.error(f"Internal server error: {exc}")
    return {
        "error": "Internal server error",
        "message": "An unexpected error occurred. Please try again later.",
        "status_code": 500
    }

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
