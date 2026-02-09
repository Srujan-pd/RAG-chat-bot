from fastapi import APIRouter, Form, Request, Response, Depends, HTTPException
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
import uuid

from database import SessionLocal
from models import Chat
from rag_engine import get_answer

router = APIRouter(prefix="/chat", tags=["Chat"])

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_or_create_session(request: Request, response: Response):
    session_id = request.cookies.get("session_id")
    if not session_id:
        session_id = str(uuid.uuid4())
        response.set_cookie(
            key="session_id",
            value=session_id,
            httponly=True,
            max_age=60 * 60 * 24 * 7,
            samesite="lax",
        )
    return session_id

@router.post("/")
async def chat_main(
    request: Request,
    response: Response,
    text: str = Form(...),
    user_id: str = Form("default_user"),
    db: Session = Depends(get_db),
):
    try:
        session_id = get_or_create_session(request, response)
        ai_text = get_answer(text)

        chat = Chat(
            session_id=session_id,
            user_id=user_id,
            question=text,
            answer=ai_text,
            created_at=datetime.utcnow(),
        )
        db.add(chat)
        db.commit()

        return {
            "status": "success",
            "message": ai_text,
            "session_id": session_id,
        }

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/history/{user_id}")
async def history(
    user_id: str,
    request: Request,
    db: Session = Depends(get_db),
    limit: int = 50,
):
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
            "id": c.id,
            "question": c.question,
            "answer": c.answer,
            "created_at": c.created_at.isoformat(),
        }
        for c in chats
    ]

