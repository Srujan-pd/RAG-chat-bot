import os
import logging
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session
from typing import List, Optional
import uvicorn

# Import modules
from database import engine, get_db, Base
from models import Chat
import rag_engine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Primis Digital Chatbot",
    description="RAG-powered chatbot with Gemini AI",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class ChatRequest(BaseModel):
    user_id: str
    session_id: str
    question: str

class ChatResponse(BaseModel):
    answer: str
    session_id: str

class VoiceRequest(BaseModel):
    user_id: str
    session_id: str
    audio_data: str  # base64 encoded

# Startup event
@app.on_event("startup")
async def startup():
    """Initialize database and load vector store"""
    logger.info("🚀 Starting Primis Digital Chatbot...")
    
    # Create tables
    Base.metadata.create_all(bind=engine)
    logger.info("✅ Database initialized")
    
    # Initialize Gemini client
    logger.info("🔑 Initializing Gemini client...")
    rag_engine.initialize_gemini()
    
    # Start vector store loading in background
    logger.info("🔄 Vector store loading started in background")
    rag_engine.start_loading_vectorstore()

# ROOT ENDPOINT
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint - Health check"""
    return {
        "status": "online",
        "service": "Primis Digital Chatbot",
        "version": "1.0.0",
        "vectorstore_loaded": not rag_engine.is_loading
    }

# HEALTH ENDPOINT
@app.get("/health", tags=["Health"])
async def health():
    """Detailed health check"""
    return {
        "status": "healthy",
        "vectorstore_status": "loaded" if not rag_engine.is_loading else "loading",
        "gemini_initialized": rag_engine.gemini_client is not None
    }

# CHAT ENDPOINT
@app.post("/chat/", response_model=ChatResponse, tags=["Chat"])
async def chat_main(request: ChatRequest, db: Session = Depends(get_db)):
    """Main chat endpoint with RAG"""
    try:
        logger.info(f"📨 Question from {request.user_id}: {request.question}")
        
        # Check if vector store is loaded
        if rag_engine.is_loading:
            logger.warning("⏳ Vector store still loading...")
            return ChatResponse(
                answer="System is initializing. Please try again in a few seconds.",
                session_id=request.session_id
            )
        
        # Get answer from RAG engine
        answer = rag_engine.get_answer(request.question)
        
        logger.info(f"✅ Answer generated: {answer[:100]}...")
        
        # Save to database
        chat_entry = Chat(
            user_id=request.user_id,
            session_id=request.session_id,
            question=request.question,
            answer=answer
        )
        db.add(chat_entry)
        db.commit()
        
        return ChatResponse(
            answer=answer,
            session_id=request.session_id
        )
        
    except Exception as e:
        logger.error(f"❌ Chat Error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# GET CHAT HISTORY
@app.get("/chat/history/{user_id}", tags=["Chat"])
async def get_chat_history(user_id: str, db: Session = Depends(get_db)):
    """Get chat history for a user"""
    try:
        chats = db.query(Chat).filter(Chat.user_id == user_id).order_by(Chat.created_at.desc()).limit(50).all()
        
        history = [{
            "id": chat.id,
            "session_id": chat.session_id,
            "question": chat.question,
            "answer": chat.answer,
            "created_at": chat.created_at.isoformat()
        } for chat in chats]
        
        return {"user_id": user_id, "history": history, "count": len(history)}
        
    except Exception as e:
        logger.error(f"❌ History Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching history: {str(e)}")

# VOICE CHAT ENDPOINT
@app.post("/voice/", tags=["Voice"])
async def voice_chat(request: VoiceRequest):
    """Voice chat endpoint (placeholder for future implementation)"""
    return {
        "message": "Voice chat endpoint - Coming soon",
        "user_id": request.user_id,
        "session_id": request.session_id
    }

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
