from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
import logging
from chat import router as chat_router
from voice_chat import router as voice_router
from fastapi.staticfiles import StaticFiles
from rag_engine import start_loading_vectorstore, initialize_gemini

app = FastAPI(title="Support AI Bot")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CORS middleware (allows frontend to communicate with backend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # CHANGE in production to specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup_event():
    print("🚀 Starting RAG system...")
    initialize_gemini()
    start_loading_vectorstore()

app.include_router(chat_router)


@app.on_event("startup")
async def startup_event():
    try:
        from database import engine, Base
        import models
        Base.metadata.create_all(bind=engine)
        logger.info("✓ Database tables synchronized")
    except Exception as e:
        logger.error(f"⚠ DB initialization failed: {e}")

# Include routers BEFORE static files to avoid conflicts
app.include_router(chat_router)
app.include_router(voice_router)
# Root redirect to static page
@app.get("/")
def root():
    return RedirectResponse(url="/static/index.html")

# Mount static files LAST
app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9000)

