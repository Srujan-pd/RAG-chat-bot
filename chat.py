from datetime import datetime, timedelta
import os
import uuid
from fastapi import APIRouter, Form, HTTPException, Depends, Request, Response
from sqlalchemy.orm import Session
from database import SessionLocal
from models import Chat
from google import genai

router = APIRouter()

# Initialize Gemini client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_or_create_session(request: Request, response: Response):
    """Get existing session ID or create a new one"""
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

def build_prompt(db, session_id, user_message, context_limit=50):
    """
    Build prompt with recent chat history for context.
    """
    chats = (
        db.query(Chat)
        .filter(Chat.session_id == session_id)
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
        prompt = f"You are a helpful support AI assistant.\n\nUser: {user_message}\nAssistant:"

    return prompt

@router.post("/chat/")
async def chat_main_chat(
    request: Request,
    response: Response,
    text: str = Form(...),
    user_id: str = Form("default_user"),
    db: Session = Depends(get_db)
):
    """
    Main chat endpoint - accepts form data
    
    Parameters:
    - text: User's message (required)
    - user_id: User identifier (default: "default_user")
    
    Returns:
    - message: AI response
    - session_id: Session identifier
    - status: Request status
    """
    try:
        session_id = get_or_create_session(request, response)

        # Build prompt with chat history
        prompt = build_prompt(db, session_id, text)

        # Get AI response from Gemini
        gemini_response = client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=prompt
        )

        ai_text = gemini_response.text

        # Save to database
        new_chat = Chat(
            session_id=session_id,
            user_id=user_id,
            question=text,
            answer=ai_text
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
        print(f"❌ Chat Error: {str(e)}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"AI generation failed: {str(e)}")


@router.get("/chat/history")
async def get_chat_history(
    request: Request,
    db: Session = Depends(get_db)
):
    """
    Get chat history for current session
    
    Returns list of chat messages from the current session (last 7 days)
    """
    session_id = request.cookies.get("session_id")
    if not session_id:
        return []
    
    seven_days_ago = datetime.utcnow() - timedelta(days=7)

    # Get messages from last 7 days
    chats = (
        db.query(Chat)
        .filter(
            Chat.session_id == session_id,
            Chat.created_at >= seven_days_ago
        )
        .order_by(Chat.created_at.desc())
        .all()
    )
    
    # Reverse to get chronological order (oldest to newest)
    chats_list = [
        {
            "id": chat.id,
            "question": chat.question,
            "answer": chat.answer,
            "created_at": chat.created_at.isoformat()
        }
        for chat in reversed(chats)
    ]
    
    return chats_list
