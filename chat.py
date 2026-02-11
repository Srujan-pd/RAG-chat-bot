from datetime import datetime, timedelta
import os
import uuid
from fastapi import APIRouter, Form, HTTPException, Depends, Request, Response
from sqlalchemy.orm import Session
from database import get_db  # Import from database, not SessionLocal
from models import Chat
from rag_engine import get_answer
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat")

# SESSION HANDLER
def get_or_create_session(request: Request, response: Response):
    session_id = request.cookies.get("session_id")
    if not session_id:
        session_id = str(uuid.uuid4())
        response.set_cookie(
            key="session_id",
            value=session_id,
            httponly=True,
            max_age=60 * 60 * 24 * 7,  # 7 days
            samesite="lax",
            secure=True  # Enable for HTTPS
        )
        logger.info(f"🆕 Created new session: {session_id}")
    return session_id

@router.post("/")
async def chat_main(
    request: Request,
    response: Response,
    text: str = Form(...),
    user_id: str = Form("default_user"),
    db: Session = Depends(get_db)
):
    """
    Main chat endpoint - accepts form data
    Uses RAG to answer from website content
    """
    try:
        session_id = get_or_create_session(request, response)
        logger.info(f"💬 Chat request - Session: {session_id}, Question: {text[:50]}...")

        # Get AI response using RAG
        ai_text = get_answer(
            question=text,
            session_id=session_id,
            db_session=db
        )

        # Save to database
        new_chat = Chat(
            session_id=session_id,
            user_id=user_id,
            question=text,
            answer=ai_text,
            created_at=datetime.utcnow()
        )
        db.add(new_chat)
        db.commit()
        db.refresh(new_chat)

        return {
            "message": ai_text,
            "session_id": session_id,
            "status": "success",
            "chat_id": new_chat.id
        }

    except Exception as e:
        logger.error(f"❌ Chat error: {str(e)}")
        db.rollback()
        
        # Fallback response
        fallback = "I'm experiencing a technical issue. Please contact us at contact@primisdigital.com"
        
        return {
            "message": fallback,
            "status": "error",
            "detail": str(e) if os.getenv("DEBUG") else "Service temporarily unavailable"
        }

@router.get("/history/{user_id}")
async def get_chat_history(
    user_id: str,
    request: Request,
    db: Session = Depends(get_db),
    limit: int = 50
):
    """
    Get chat history for current session
    """
    session_id = request.cookies.get("session_id")
    if not session_id:
        logger.info("📋 No session ID for history request")
        return []

    seven_days_ago = datetime.utcnow() - timedelta(days=7)

    try:
        # Get last N messages in chronological order
        chats = (
            db.query(Chat)
            .filter(
                Chat.session_id == session_id,
                Chat.created_at >= seven_days_ago
            )
            .order_by(Chat.created_at.asc())
            .limit(limit)
            .all()
        )

        logger.info(f"📋 Retrieved {len(chats)} messages for session {session_id}")

        return [
            {
                "id": chat.id,
                "question": chat.question,
                "answer": chat.answer,
                "created_at": chat.created_at.isoformat() if chat.created_at else None,
                "session_id": chat.session_id
            }
            for chat in chats
        ]

    except Exception as e:
        logger.error(f"❌ History Error: {str(e)}")
        return []
