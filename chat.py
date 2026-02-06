from datetime import datetime, timedelta
import os
import uuid
from fastapi import APIRouter, Form, HTTPException, Depends, Request, Response
from sqlalchemy.orm import Session
from database import get_db
from models import Chat
from rag_engine import get_answer

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
            samesite="lax"
        )
    return session_id

# LOAD CHAT HISTORY FOR CONTEXT
def build_prompt(db, session_id, user_message, context_limit=10):
    """
    Build prompt with recent chat history for context.
    context_limit: Number of recent exchanges to include (default 10)
    """
    seven_days_ago = datetime.utcnow() - timedelta(days=7)

    chats = (
        db.query(Chat)
        .filter(
            Chat.session_id == session_id,
            Chat.created_at >= seven_days_ago
        )
        .order_by(Chat.created_at.desc())
        .limit(context_limit)
        .all()
    )

    # Reverse to get chronological order
    chats = list(reversed(chats))

    if chats:
        prompt = "You are a helpful support AI assistant. Previous conversation:\n\n"
        for chat in chats:
            prompt += f"User: {chat.question}\n"
            prompt += f"Assistant: {chat.answer}\n\n"
        prompt += f"User: {user_message}\nAssistant:"
    else:
        # First message - add greeting context
        if user_message.lower().strip() in ["hi", "hello", "hey", "hi there", "hello there", "hey there"]:
            prompt = """You are a friendly AI assistant for Primis Digital. Start with a warm greeting and introduce yourself briefly. Then ask how you can help.

User: {user_message}
Assistant:""".format(user_message=user_message)
        else:
            prompt = f"""You are a helpful AI assistant for Primis Digital. You provide information about Primis Digital's services, technologies, and projects.

User: {user_message}
Assistant:"""

    return prompt

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

        # Get AI response using RAG
        ai_text = get_answer(text)

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

    except Exception as e:
        print("❌ Error:", str(e))
        db.rollback()
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
        print("❌ History Error:", str(e))
        return []
