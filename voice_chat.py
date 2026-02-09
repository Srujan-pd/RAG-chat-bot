import os
import uuid
import base64
from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException
from sqlalchemy.orm import Session
from database import SessionLocal
from models import Chat

# CORRECT IMPORT for google-generativeai package
import google.generativeai as genai

# Only import TTS if available
try:
    from google.cloud import texttospeech
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    print("⚠️ Warning: Google Cloud Text-to-Speech not available")

router = APIRouter()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Initialize Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.0-flash-exp")

tts_client = None

def get_tts_client():
    """Initialize TTS client lazily"""
    global tts_client
    if not TTS_AVAILABLE:
        raise HTTPException(
            status_code=503, 
            detail="Text-to-Speech service not available"
        )
    
    if tts_client is None:
        try:
            tts_client = texttospeech.TextToSpeechClient()
        except Exception as e:
            raise HTTPException(
                status_code=500, 
                detail=f"TTS client initialization failed: {str(e)}"
            )
    return tts_client

@router.post("/voice/")
async def voice_chat(
    file: UploadFile = File(..., description="Audio file (WAV, MP3, etc.)"),
    user_id: str = Form(default="default_user", description="User identifier"),
    db: Session = Depends(get_db)
):
    """
    Voice chat endpoint - processes audio input and returns AI response
    
    Parameters:
    - file: Audio file containing user's speech
    - user_id: User identifier (default: "default_user")
    
    Returns:
    - user_said: Transcribed user speech
    - message: AI text response
    - audio: Base64-encoded audio response (if TTS available)
    - status: Request status
    """
    audio_bytes = await file.read()
    
    # Check if audio file is too small (likely silence)
    if len(audio_bytes) < 1000:
        return {
            "user_said": "",
            "message": "No speech detected. Please speak clearly.",
            "audio": None,
            "status": "no_speech"
        }
    
    try:
        # Use Gemini to transcribe and respond to audio using multimodal capabilities
        # Note: For Gemini 2.0, we need to use the correct upload method
        prompt = """You are analyzing audio input. Your task is to:
1. Transcribe what the user said
2. Provide a helpful support response

Format your response EXACTLY like this:
USER_SAID: [transcribed text here]
AI_RESPONSE: [your helpful response here]

If there is no clear speech or only silence/noise, respond with:
SILENCE_DETECTED"""

        # Upload audio file to Gemini
        audio_file = genai.upload_file(
            path=file.filename if hasattr(file, 'filename') else "audio.wav",
            mime_type=file.content_type
        )
        
        # Generate response with audio
        gemini_response = model.generate_content([prompt, audio_file])
        full_text = gemini_response.text.strip()
        
        # Check for silence detection
        if "SILENCE_DETECTED" in full_text or full_text.upper() == "SILENCE_DETECTED":
            return {
                "user_said": "",
                "message": "I couldn't hear you clearly. Please try again.",
                "audio": None,
                "status": "no_speech"
            }
        
        # Parse the response
        if "USER_SAID:" in full_text and "AI_RESPONSE:" in full_text:
            user_text = full_text.split("USER_SAID:")[1].split("AI_RESPONSE:")[0].strip()
            ai_text = full_text.split("AI_RESPONSE:")[1].strip()
            
            # Check if transcription is valid
            if len(user_text) < 3 or user_text.upper() in ["EMPTY", "UNCLEAR", "NOISE", "...", "N/A"]:
                return {
                    "user_said": "",
                    "message": "I couldn't understand what you said. Please speak clearly.",
                    "audio": None,
                    "status": "unclear_speech"
                }
        else:
            return {
                "user_said": "",
                "message": "I couldn't process your audio. Please try again.",
                "audio": None,
                "status": "processing_error"
            }

        # Generate TTS audio if available
        audio_base64 = None
        if TTS_AVAILABLE:
            try:
                synthesis_input = texttospeech.SynthesisInput(text=ai_text)
                voice = texttospeech.VoiceSelectionParams(
                    language_code="en-US", 
                    ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
                )
                audio_config = texttospeech.AudioConfig(
                    audio_encoding=texttospeech.AudioEncoding.MP3
                )
                
                tts_client_instance = get_tts_client()
                tts_res = tts_client_instance.synthesize_speech(
                    input=synthesis_input, 
                    voice=voice, 
                    audio_config=audio_config
                )
                audio_base64 = base64.b64encode(tts_res.audio_content).decode('utf-8')
            except Exception as e:
                print(f"⚠️ TTS generation failed: {e}")

        # Save to database
        db.add(Chat(
            user_id=user_id, 
            session_id=str(uuid.uuid4()),
            question=user_text, 
            answer=ai_text
        ))
        db.commit()

        return {
            "user_said": user_text,
            "message": ai_text,
            "audio": audio_base64,
            "status": "success"
        }
        
    except Exception as e:
        print(f"❌ Voice processing error: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Voice processing error: {str(e)}"
        )
