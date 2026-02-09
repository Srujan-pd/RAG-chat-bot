from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
import logging
import datetime
import time
import sys
from contextlib import asynccontextmanager
from sqlalchemy import text

from chat import router as chat_router
from voice_chat import router as voice_router
from rag_engine import init_rag, is_rag_initialized, init_error

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
    "startup_errors": [],
    "initialization_complete": False
}

def init_database_with_retry():
    """Initialize database with retry logic"""
    try:
        from database import init_database, Base
        import models  # noqa - to ensure models are registered
        
        # Initialize database with retry
        db_success = init_database(max_retries=5, retry_delay=2)
        if db_success:
            # Get engine after initialization
            from database import engine
            if engine:
                Base.metadata.create_all(bind=engine)
                return True, "Database initialized successfully"
            else:
                return False, "Database engine is None after initialization"
        else:
            return False, "Database initialization failed"
            
    except Exception as e:
        return False, f"Database initialization failed: {str(e)}"

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events
    """
    # Startup
    logger.info("🚀 Starting Primis Digital Chatbot...")
    app_state["startup_time"] = datetime.datetime.utcnow()
    
    # Database initialization (blocking)
    try:
        logger.info("=" * 60)
        logger.info("STEP 1: Initializing database...")
        logger.info("=" * 60)
        db_success, db_message = init_database_with_retry()
        if db_success:
            app_state["database_ready"] = True
            logger.info(f"✅ {db_message}")
        else:
            app_state["startup_errors"].append(db_message)
            logger.error(f"❌ {db_message}")
            # Continue without database - app can still run in degraded mode
            
    except Exception as e:
        error_msg = f"Database initialization error: {str(e)}"
        app_state["startup_errors"].append(error_msg)
        logger.exception("❌ Database initialization failed")
    
    # RAG initialization (BLOCKING - wait for it to complete)
    try:
        logger.info("=" * 60)
        logger.info("STEP 2: Initializing RAG system...")
        logger.info("=" * 60)
        
        rag_start = time.time()
        rag_success = init_rag()
        rag_time = time.time() - rag_start
        
        if rag_success:
            app_state["rag_ready"] = True
            logger.info(f"✅ RAG system ready in {rag_time:.2f} seconds")
        else:
            error_msg = f"RAG initialization failed: {init_error}"
            app_state["startup_errors"].append(error_msg)
            logger.error(f"❌ {error_msg}")
            # This is critical - exit if RAG fails
            logger.error("RAG system is required. Exiting...")
            sys.exit(1)
            
    except Exception as e:
        error_msg = f"RAG initialization error: {str(e)}"
        app_state["startup_errors"].append(error_msg)
        logger.exception("❌ RAG initialization failed")
        sys.exit(1)
    
    app_state["initialization_complete"] = True
    total_time = time.time() - app_state["startup_time"].timestamp()
    
    logger.info("=" * 60)
    logger.info(f"✅ Application startup completed in {total_time:.2f} seconds")
    logger.info("Ready to accept requests")
    logger.info("=" * 60)
    
    yield
    
    # Shutdown
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
    from rag_engine import vectorstore, gemini_client
    
    return {
        "service": "Primis Digital Chatbot API",
        "status": "online",
        "version": "1.0.0",
        "uptime": str(datetime.datetime.utcnow() - app_state["startup_time"]) if app_state["startup_time"] else "unknown",
        "components": {
            "database": app_state["database_ready"],
            "rag_initialized": is_rag_initialized,
            "vectorstore_loaded": vectorstore is not None,
            "gemini_ready": gemini_client is not None,
            "initialization_complete": app_state["initialization_complete"]
        },
        "endpoints": {
            "health": "/health",
            "ready": "/ready",
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
                result = conn.execute(text("SELECT 1"))
                result.fetchone()
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
    if vectorstore is not None:
        health_status["components"]["vectorstore"] = {
            "status": "loaded",
            "message": "OK"
        }
    else:
        health_status["components"]["vectorstore"] = {
            "status": "error",
            "message": init_error[:100] if init_error else "Vector store not loaded"
        }
        health_status["status"] = "unhealthy"
    
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
        health_status["status"] = "unhealthy"
    
    # Add startup errors if any
    if app_state["startup_errors"]:
        health_status["startup_errors"] = app_state["startup_errors"]
        if health_status["status"] == "healthy":
            health_status["status"] = "degraded"
    
    return health_status

# -----------------------------
# READINESS PROBE (for Cloud Run)
# -----------------------------
@app.get("/ready")
def ready():
    """Readiness probe - checks if service is ready to accept traffic"""
    from rag_engine import vectorstore, gemini_client, is_rag_initialized
    
    # Service is ready ONLY if both database AND RAG are ready
    try:
        from database import engine
        
        # Check database
        db_ready = False
        if engine is not None:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            db_ready = True
        
        # Check RAG
        rag_ready = (
            is_rag_initialized and 
            vectorstore is not None and 
            gemini_client is not None
        )
        
        if db_ready and rag_ready:
            return {
                "ready": True,
                "message": "Service is ready to accept requests",
                "components": {
                    "database": "ready",
                    "rag": "ready",
                    "vectorstore": "loaded",
                    "gemini": "initialized"
                }
            }
        else:
            return {
                "ready": False,
                "reason": "Service is still initializing",
                "components": {
                    "database": "ready" if db_ready else "not ready",
                    "rag": "ready" if rag_ready else "initializing",
                    "vectorstore": "loaded" if vectorstore is not None else "loading",
                    "gemini": "initialized" if gemini_client is not None else "initializing"
                }
            }
        
    except Exception as e:
        return {
            "ready": False,
            "reason": f"Error: {str(e)[:100]}"
        }

# -----------------------------
# LIVENESS PROBE
# -----------------------------
@app.get("/alive")
def alive():
    """Liveness probe - checks if service is running"""
    return {
        "alive": True,
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "service": "Primis Digital Chatbot API",
        "uptime": str(datetime.datetime.utcnow() - app_state["startup_time"]) if app_state["startup_time"] else "unknown"
    }

# -----------------------------
# DATABASE STATUS ENDPOINT
# -----------------------------
@app.get("/db-status")
def db_status():
    """Check database status"""
    try:
        from database import engine, _initialized
        if engine is None:
            return {
                "initialized": _initialized,
                "engine": "None",
                "status": "not_initialized"
            }
        
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1")).fetchone()
        
        return {
            "initialized": _initialized,
            "engine": "available",
            "connection_test": result[0] if result else None,
            "status": "connected"
        }
    except Exception as e:
        return {
            "initialized": False,
            "engine": "error",
            "error": str(e),
            "status": "error"
        }

# -----------------------------
# RAG STATUS ENDPOINT
# -----------------------------
@app.get("/rag-status")
def rag_status():
    """Check RAG system status"""
    from rag_engine import vectorstore, gemini_client, init_error, is_rag_initialized
    
    return {
        "rag_initialized": is_rag_initialized,
        "vectorstore_loaded": vectorstore is not None,
        "gemini_initialized": gemini_client is not None,
        "init_error": init_error,
        "rag_ready": app_state["rag_ready"],
        "initialization_complete": app_state["initialization_complete"]
    }

# -----------------------------
# ERROR HANDLERS
# -----------------------------
@app.exception_handler(500)
async def internal_server_error_handler(request: Request, exc: Exception):
    """Handle 500 errors gracefully"""
    logger.error(f"Internal server error: {exc}")
    return Response(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred. Please try again later.",
            "status_code": 500
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {exc}")
    return Response(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc),
            "status_code": 500
        }
    )

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")

