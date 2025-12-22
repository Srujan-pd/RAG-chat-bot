from fastapi import APIRouter, Depends, Form, HTTPException, Request
import logging

logger = logging.getLogger(__name__)
from sqlalchemy.orm import Session
from database import SessionLocal
from models import Chat
import google.generativeai as genai
import os
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


# per-user in-memory histories
chat_histories = {}


# ✅ FIXED MODEL
model = genai.GenerativeModel("models/gemini-flash-latest")

router = APIRouter(prefix="/chat")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/")
async def chat_form(request: Request, user_id: int = Form(None), prompt: str = Form(None), db: Session = Depends(get_db)):
    """Legacy root endpoint that accepts either form-encoded fields (`user_id`, `prompt`) or JSON body
    (`user_id` and either `prompt` or `message`). This avoids 422 errors when clients send JSON to `/`.
    """
    # If form data not provided, try to read JSON body
    if user_id is None or prompt is None:
        try:
            payload = await request.json()
            if user_id is None:
                user_id = payload.get("user_id")
            if prompt is None:
                # support both `prompt` and `message` keys
                prompt = payload.get("prompt") or payload.get("message")
        except Exception:
            # could not read JSON, continue to validation below
            pass

    # validate - build a list of missing fields only
    errors = []
    if user_id is None:
        errors.append({"type": "missing", "loc": ["body", "user_id"], "msg": "Field required", "input": user_id})
    if prompt is None:
        errors.append({"type": "missing", "loc": ["body", "prompt"], "msg": "Field required", "input": prompt})
    if errors:
        raise HTTPException(status_code=422, detail=errors)

    try:
        response = model.generate_content(prompt)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    chat = Chat(
        user_id=user_id,
        question=prompt,
        answer=response.text
    )
    db.add(chat)
    db.commit()

    return {
        "question": prompt,
        "answer": response.text
    }


# static rule-based responses (exact-match, normalized)
static_responses = {
    "how are you": "I am functioning well and ready to assist you! Thank you for asking.\n\nHow can I help you today?"
}


def _normalize_text(s: str) -> str:
    """Simple normalization for comparisons: lower-case and strip punctuation/extra whitespace."""
    import re
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9\s]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def get_bot_response(user_message: str, user_id: int = 1):
    """Uses a per-user history list to generate a contextual reply.

    First check static rule-based responses; if none match, call the model and store the exchange.
    """
    # ensure history for this user
    if user_id not in chat_histories:
        chat_histories[user_id] = []

    # check static responses (exact match after normalization)
    normalized = _normalize_text(user_message)
    if normalized in static_responses:
        # append user message
        chat_histories[user_id].append({
            "role": "user",
            "parts": [user_message]
        })

        bot_reply = static_responses[normalized]

        # append bot reply to history
        chat_histories[user_id].append({
            "role": "model",
            "parts": [bot_reply]
        })

        return bot_reply

    # append user message
    chat_histories[user_id].append({
        "role": "user",
        "parts": [user_message]
    })

    # call model with the conversation so far
    response = model.generate_content(chat_histories[user_id])

    # bot reply
    bot_reply = response.text

    # append bot reply to history
    chat_histories[user_id].append({
        "role": "model",
        "parts": [bot_reply]
    })

    return bot_reply


class ChatRequest(BaseModel):
    message: str
    user_id: int = 1

@router.post("/chat/")
def chat(req: ChatRequest, db: Session = Depends(get_db)):
    """Primary JSON endpoint used by the frontend: returns a reply and persists the Q/A to DB.

    Wrapped in try/except with logging to make failures visible in server logs and return clear errors
    rather than silent or malformed responses.
    """
    user_id = req.user_id
    user_message = req.message

    try:
        bot_reply = get_bot_response(user_message, user_id)

        # persist to DB
        chat = Chat(
            user_id=user_id,
            question=user_message,
            answer=bot_reply
        )
        db.add(chat)
        db.commit()

        return {"reply": bot_reply}
    except Exception as e:
        logger.exception("Error handling /chat/ request")
        # return an informative 500 to the client
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history/{user_id}")
def history(user_id: int, db: Session = Depends(get_db)):
    """Return past chats for a user (ordered by time ascending)."""
    rows = db.query(Chat).filter(Chat.user_id == user_id).order_by(Chat.created_at).all()
    results = [{"question": r.question, "answer": r.answer, "created_at": r.created_at.isoformat()} for r in rows]
    return {"history": results}
