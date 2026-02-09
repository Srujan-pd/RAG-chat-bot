import os, uuid
from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException
from sqlalchemy.orm import Session
from database import get_db as get_db_dependency  # Rename import to avoid conflict
from models import Chat
from google import genai
from google.genai import types

router = APIRouter(prefix="/voice")

# Lazy client initialization
_client = None

def get_gemini_client():
    """Get or create Gemini client"""
    global _client
    if _client is None:
        # Remove GOOGLE_API_KEY to avoid conflicts
        if "GOOGLE_API_KEY" in os.environ:
            del os.environ["GOOGLE_API_KEY"]
        
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set!")
        
        _client = genai.Client(api_key=api_key)
    return _client

@router.post("/")
async def voice_chat(
    file: UploadFile = File(...), 
    user_id: str = Form("default_user"),
    db: Session = Depends(get_db_dependency)  # Use the renamed import
):
    """
    Voice chat endpoint - accepts audio file
    Transcribes and responds using Gemini
    """
    audio_bytes = await file.read()
    
    try:
        client = get_gemini_client()
        
        # Step 1: Gemini handles transcription + logic
        model_res = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[
                "Transcribe the audio and provide a helpful response. Format: [Transcription] | [Response]",
                types.Part.from_bytes(data=audio_bytes, mime_type=file.content_type)
            ]
        )
        
        full_text = model_res.text
        parts = full_text.split("|")
        user_text = parts[0].strip() if len(parts) > 1 else "Voice Message"
        ai_text = parts[-1].strip()

        # Step 2: Save to DB
        new_chat = Chat(
            user_id=user_id, 
            session_id="voice_session", 
            question=user_text, 
            answer=ai_text
        )
        db.add(new_chat)
        db.commit()

        return {
            "user_said": user_text,
            "message": ai_text,
            "status": "success"
        }
    except Exception as e:
        # Don't rollback if db is None
        if db is not None:
            try:
                db.rollback()
            except:
                pass
        raise HTTPException(status_code=500, detail=str(e))
