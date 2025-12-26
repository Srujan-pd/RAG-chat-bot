from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from database import SessionLocal
from models import Chat
import os
import google.generativeai as genai
from pydantic import BaseModel

router = APIRouter(prefix="/chat")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("models/gemini-flash-latest")
else:
    model = None

chat_histories = {}

class ChatRequest(BaseModel):
    message: str
    user_id: int

@router.post("/")
def chat(req: ChatRequest, db: Session = Depends(get_db)):
    if not model:
        return {"reply": "AI Configuration Error: API Key missing."}
    
    try:
        user_id = req.user_id
        user_message = req.message

        # Initialize history
        if user_id not in chat_histories:
            chat_histories[user_id] = []

        chat_histories[user_id].append({"role": "user", "parts": [user_message]})
        
        # Generate AI response
        response = model.generate_content(chat_histories[user_id])
        bot_reply = response.text
        
        chat_histories[user_id].append({"role": "model", "parts": [bot_reply]})

        # Save to Postgres
        new_chat = Chat(user_id=user_id, question=user_message, answer=bot_reply)
        db.add(new_chat)
        db.commit()

        return {"reply": bot_reply}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/history/{user_id}")
def history(user_id: int, db: Session = Depends(get_db)):
    try:
        rows = db.query(Chat).filter(Chat.user_id == user_id).all()
        return {"history": [{"question": r.question, "answer": r.answer} for r in rows]}
    except Exception as e:
        print(f"Database Error: {e}")
        return {"history": []}
