# chat.py - This should be a router, not a duplicate of main.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
import uuid
import logging
from database import get_db
from rag_engine import get_answer
from models import Chat

router = APIRouter(prefix="/chat")
logger = logging.getLogger(__name__)

# Request/Response models
class ChatRequest(BaseModel):
    question: str
    user_id: str = "default_user"

class ChatResponse(BaseModel):
    answer: str
    session_id: str

@router.post("/")
async def chat_endpoint(request: ChatRequest, db: Session = Depends(get_db)):
    """
    Main chat endpoint
    """
    try:
        # Generate or use session ID
        session_id = str(uuid.uuid4())
        
        # Get answer from RAG system
        answer = get_answer(request.question, session_id, db)
        
        # Save to database
        chat_record = Chat(
            session_id=session_id,
            user_id=request.user_id,
            question=request.question,
            answer=answer
        )
        db.add(chat_record)
        db.commit()
        
        return {
            "answer": answer,
            "session_id": session_id,
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        if db:
            db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status")
async def chat_status():
    """Check RAG system status"""
    from rag_engine import get_vectorstore_status
    status = get_vectorstore_status()
    return {"status": "ready" if status["loaded"] else "loading", "details": status}
