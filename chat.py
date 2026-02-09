from datetime import datetime, timedelta
import uuid
from fastapi import APIRouter, Form, HTTPException, Depends, Request, Response
from sqlalchemy.orm import Session

from database import SessionLocal
from models import Chat
from rag_engine import get_answer

router = APIRouter(prefix="/chat", tags=["Chat"])

# -------------------------
# DB dependency
# -------------------------
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# -------------------------
# Session handler
# -------------------------
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
        )
    return session_id

# -------------------------
# Chat endpoint
# -------------------------
@router.post("/")
async def chat_main(
    request: Request,
    response: Response,
    text: str = Form(...),
    user_id: str = Form("default_user"),
    db: Session = Depends(get_db),
):
    """
    Main chat endpoint
    Uses RAG to answer from website content
    """
    try:
        session_id = get_or_create_session(request, response)

        # 🔥 RAG call (vectorstore already loaded at startup)
        ai_text = get_answer(text)

        # Save chat to DB
        chat = Chat(
            session_id=session_id,
            user_id=user_id,
            question=text,
            answer=ai_text,
            created_at=datetime.utcnow(),
        )
        db.add(chat)
        db.commit()
        db.refresh(chat)

        return {
            "status": "success",
            "message": ai_text,
            "session_id": session_id,
        }

    except Exception as e:
        print("❌ Chat Error:", str(e))
        db.rollback()
        raise HTTPException(status_code=500, detail="AI generation failed")

# -------------------------
# Chat history endpoint
# -------------------------
@router.get("/history/{user_id}")
async def get_chat_history(
    user_id: str,
    request: Request,
    db: Session = Depends(get_db),
    limit: int = 50,
):
    """
    Returns chat history for current session
    """
    session_id = request.cookies.get("session_id")
    if not session_id:
        return []

    seven_days_ago = datetime.utcnow() - timedelta(days=7)

    chats = (
        db.query(Chat)
        .filter(
            Chat.session_id == session_id,
            Chat.created_at >= seven_days_ago,
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
            "created_at": chat.created_at.isoformat()
            if chat.created_at
            else None,
        }
        for chat in chats
    ]

