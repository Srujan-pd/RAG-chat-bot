from datetime import datetime, timedelta
import os, uuid
import logging
from fastapi import APIRouter, Form, HTTPException, Depends, Request, Response
from sqlalchemy.orm import Session
from database import SessionLocal
from models import Chat
from rag_engine import get_answer, is_vectorstore_ready

router = APIRouter(prefix="/chat")
logger = logging.getLogger(__name__)

def get_db():
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        logger.error(f"Database error: {e}")
        raise
    finally:
        db.close()


# SESSION HANDLER
def get_or_create_session(request: Request, response: Response):
    session_id = request.cookies.get("session_id")
    if not session_id:
        session_id = str(uuid.uuid4())
        response.set_cookie(
            key="session_id",
            value=session_id,
            httponly=True,
            max_age = 60 * 60 * 24 * 7,    # 7days
            samesite="lax"
        )
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
        # Check if vector store is ready
        if not is_vectorstore_ready():
            raise HTTPException(
                status_code=503,
                detail="Knowledge base is still loading. Please try again in a moment."
            )
        
        session_id = get_or_create_session(request, response)

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
            "status": "success"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        try:
            db.rollback()
        except:
            pass
        raise HTTPException(status_code=500, detail=f"AI generation failed: {str(e)}")


@router.get("/history/{user_id}")
async def get_chat_history(
    user_id: str,
    request: Request,
    db: Session = Depends(get_db),
    limit: int = 50
):
    """
    Get chat history for current session
    Returns array directly so frontend .slice() works
    """
    session_id = request.cookies.get("session_id")
    if not session_id:
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

        return [
            {
                "id": chat.id,
                "question": chat.question,
                "answer": chat.answer,
                "created_at": chat.created_at.isoformat() if chat.created_at else None
            }
            for chat in chats
        ]

    except Exception as e:
        logger.error(f"History error: {str(e)}")
        return []
