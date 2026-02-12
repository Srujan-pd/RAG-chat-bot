import os, uuid
from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException, Request, Response as FastAPIResponse
from fastapi.responses import Response
from sqlalchemy.orm import Session
from database import SessionLocal
from models import Chat
from google import genai
from google.genai import types
from google.cloud import texttospeech
from rag_engine import get_answer  # Import RAG engine
from datetime import datetime
import logging
import traceback

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/voice")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Lazy client initialization
_gemini_client = None
_tts_client = None

def get_gemini_client():
    """Get or create Gemini client for transcription"""
    global _gemini_client
    if _gemini_client is None:
        # Remove GOOGLE_API_KEY to avoid conflicts
        if "GOOGLE_API_KEY" in os.environ:
            del os.environ["GOOGLE_API_KEY"]
        
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set!")
        
        _gemini_client = genai.Client(api_key=api_key)
        logger.info("✅ Gemini client initialized for voice transcription")
    return _gemini_client

def get_tts_client():
    """Get or create Text-to-Speech client"""
    global _tts_client
    if _tts_client is None:
        try:
            # Check if running in Cloud Run with service account
            if os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
                logger.info("🔐 Using GOOGLE_APPLICATION_CREDENTIALS for TTS")
            elif os.getenv("GOOGLE_CLOUD_PROJECT"):
                logger.info("☁️ Using default Cloud Run service account for TTS")
            else:
                logger.warning("⚠️ No Google Cloud credentials found - TTS may not work")
            
            _tts_client = texttospeech.TextToSpeechClient()
            logger.info("✅ Text-to-Speech client initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize TTS client: {str(e)}")
            logger.error(traceback.format_exc())
            # Don't raise - let it return None so fallback works
            logger.warning("⚠️ TTS will be disabled - using text-only responses")
            _tts_client = None
    return _tts_client

def text_to_speech(text: str) -> bytes:
    """Convert text to speech audio using Google Cloud TTS"""
    try:
        logger.info(f"🔊 Generating TTS for text: {text[:100]}...")
        client = get_tts_client()
        
        # If client initialization failed, return None
        if client is None:
            logger.warning("⚠️ TTS client not available - returning None")
            return None
        
        # Truncate text if too long (TTS has limits)
        max_chars = 5000
        if len(text) > max_chars:
            text = text[:max_chars] + "..."
            logger.warning(f"⚠️ Text truncated to {max_chars} characters for TTS")
        
        # Configure synthesis input
        synthesis_input = texttospeech.SynthesisInput(text=text)
        
        # Configure voice parameters
        voice = texttospeech.VoiceSelectionParams(
            language_code="en-US",
            name="en-US-Neural2-F",  # Female voice
            ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
        )
        
        # Configure audio settings
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=1.0,
            pitch=0.0
        )
        
        # Generate speech
        response = client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config
        )
        
        logger.info(f"✅ TTS generated successfully, size: {len(response.audio_content)} bytes")
        return response.audio_content
        
    except Exception as e:
        logger.error(f"❌ TTS error: {str(e)}")
        logger.error(traceback.format_exc())
        # Fallback: return None if TTS fails
        return None

# SESSION HANDLER (same as in chat.py)
def get_or_create_session(request: Request, response: FastAPIResponse):
    session_id = request.cookies.get("session_id")
    if not session_id:
        session_id = str(uuid.uuid4())
        response.set_cookie(
            key="session_id",
            value=session_id,
            httponly=True,
            max_age = 60 * 60 * 24 * 7,    # 7 days
            samesite="lax"
        )
    return session_id

@router.post("/")
async def voice_chat(
    request: Request,
    response: FastAPIResponse,
    file: UploadFile = File(...), 
    user_id: str = Form("default_user"),
    response_format: str = Form("json"),  # "json" or "audio"
    db: Session = Depends(get_db)
):
    """
    Voice chat endpoint with RAG - accepts audio file, returns text or audio
    
    Parameters:
    - file: Audio file (webm, mp3, wav, etc.)
    - user_id: User identifier
    - response_format: "json" for text response, "audio" for TTS audio
    
    Returns:
    - JSON with transcription and text response (if response_format=json)
    - MP3 audio file (if response_format=audio)
    
    Features:
    - Uses RAG to answer from Primis Digital knowledge base
    - Supports session-based conversation history
    - Text-to-Speech for audio responses
    """
    logger.info(f"📞 Voice chat request from user: {user_id}, format: {response_format}")
    
    try:
        audio_bytes = await file.read()
        logger.info(f"📁 Received audio file: {file.filename}, size: {len(audio_bytes)} bytes, type: {file.content_type}")
        
        # Get or create session
        session_id = get_or_create_session(request, response)
        logger.info(f"🔑 Session ID: {session_id}")
        
        # Step 1: Transcribe audio using Gemini
        logger.info("🎤 Transcribing audio...")
        client = get_gemini_client()
        
        # Use Gemini to transcribe the audio
        transcription_res = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[
                "Transcribe this audio exactly as spoken. Only return the transcription text, nothing else.",
                types.Part.from_bytes(data=audio_bytes, mime_type=file.content_type or "audio/webm")
            ]
        )
        
        user_text = transcription_res.text.strip()
        logger.info(f"✅ Transcription: {user_text}")
        
        if not user_text:
            raise ValueError("Transcription failed - no text returned")
        
        # Step 2: Use RAG to get answer (same as text chat)
        logger.info("🤖 Getting RAG answer...")
        ai_text = get_answer(
            question=user_text,
            session_id=session_id,
            db_session=db
        )
        logger.info(f"✅ RAG answer: {ai_text[:100]}...")

        # Step 3: Save to DB
        logger.info("💾 Saving to database...")
        new_chat = Chat(
            user_id=user_id, 
            session_id=session_id,  # Use actual session ID instead of "voice_session"
            question=user_text, 
            answer=ai_text,
            created_at=datetime.utcnow()
        )
        db.add(new_chat)
        db.commit()
        db.refresh(new_chat)
        logger.info(f"✅ Chat saved with ID: {new_chat.id}")

        # Step 4: Return based on format
        if response_format == "audio":
            logger.info("🔊 Generating audio response...")
            # Generate TTS audio
            audio_content = text_to_speech(ai_text)
            
            if audio_content:
                logger.info("✅ Returning audio response")
                return Response(
                    content=audio_content,
                    media_type="audio/mpeg",
                    headers={
                        "Content-Disposition": "attachment; filename=response.mp3",
                        "Content-Length": str(len(audio_content))
                    }
                )
            else:
                # Fallback to JSON if TTS fails
                logger.warning("⚠️ TTS failed, returning JSON response")
                return {
                    "user_said": user_text,
                    "message": ai_text,
                    "session_id": session_id,
                    "status": "success",
                    "error": "TTS generation failed, returning text"
                }
        else:
            # Return JSON response
            logger.info("✅ Returning JSON response")
            return {
                "user_said": user_text,
                "message": ai_text,
                "session_id": session_id,
                "status": "success"
            }
            
    except Exception as e:
        logger.error(f"❌ Voice chat error: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Voice chat failed: {str(e)}")

