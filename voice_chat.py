import os
import uuid
import io
from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException, Response
from sqlalchemy.orm import Session
from database import SessionLocal, get_db
from models import Chat
from google import genai
from google.genai import types
from gtts import gTTS
import logging

logger = logging.getLogger(__name__)

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
        logger.info("✅ Gemini client initialized for voice")
    return _client

@router.post("/")
async def voice_chat(
    file: UploadFile = File(...), 
    user_id: str = Form("default_user"),
    db: Session = Depends(get_db),
    return_audio: bool = Form(True)
):
    """
    Voice chat endpoint - accepts audio file
    Transcribes, responds, and returns audio response
    """
    audio_bytes = await file.read()
    
    try:
        client = get_gemini_client()
        
        # Enhanced prompt for voice responses
        model_res = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[
                "You are Primis Digital's voice assistant. Provide a clear, concise, and conversational response. "
                "Be helpful and friendly. If asked about contact information, always include phone and email if available. "
                "If you don't know something, suggest contacting Primis Digital directly at contact@primisdigital.com.",
                types.Part.from_bytes(data=audio_bytes, mime_type=file.content_type or "audio/wav")
            ],
            config=types.GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=300
            )
        )
        
        ai_text = model_res.text.strip()
        logger.info(f"🎤 Voice input processed, response: {ai_text[:100]}...")

        # Generate a session ID if not present
        session_id = f"voice_{uuid.uuid4()}"

        # Save to database
        new_chat = Chat(
            session_id=session_id,
            user_id=user_id, 
            question="[Voice Input]", 
            answer=ai_text,
            created_at=datetime.utcnow()
        )
        db.add(new_chat)
        db.commit()

        # Generate TTS response
        if return_audio:
            tts = gTTS(text=ai_text, lang='en', slow=False)
            audio_fp = io.BytesIO()
            tts.write_to_fp(audio_fp)
            audio_fp.seek(0)
            
            return Response(
                content=audio_fp.read(),
                media_type="audio/mpeg",
                headers={
                    "X-Response-Text": ai_text,
                    "X-Session-ID": session_id,
                    "Access-Control-Expose-Headers": "X-Response-Text, X-Session-ID"
                }
            )
        
        return {
            "message": ai_text,
            "session_id": session_id,
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"❌ Voice chat error: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Fallback response
        fallback_text = "I'm having trouble processing your voice request. Please try again or contact us at contact@primisdigital.com"
        
        if return_audio:
            tts = gTTS(text=fallback_text, lang='en', slow=False)
            audio_fp = io.BytesIO()
            tts.write_to_fp(audio_fp)
            audio_fp.seek(0)
            
            return Response(
                content=audio_fp.read(),
                media_type="audio/mpeg",
                headers={
                    "X-Response-Text": fallback_text,
                    "X-Error": "true"
                }
            )
        
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/tts")
async def text_to_speech(text: str = Form(...)):
    """
    Convert text to speech
    """
    try:
        tts = gTTS(text=text, lang='en', slow=False)
        audio_fp = io.BytesIO()
        tts.write_to_fp(audio_fp)
        audio_fp.seek(0)
        
        return Response(
            content=audio_fp.read(),
            media_type="audio/mpeg",
            headers={
                "Content-Disposition": f"attachment; filename=speech.mp3"
            }
        )
    except Exception as e:
        logger.error(f"❌ TTS error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/test")
async def voice_test():
    """Test endpoint for voice service"""
    return {
        "status": "Voice service is running",
        "tts_available": True,
        "gemini_configured": bool(os.getenv("GEMINI_API_KEY"))
    }
