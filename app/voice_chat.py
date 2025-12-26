from google.cloud import speech, texttospeech
import google.generativeai as genai
import uuid, os

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

model = genai.GenerativeModel("models/gemini-flash-latest")

speech_client = speech.SpeechClient()
tts_client = texttospeech.TextToSpeechClient()

chat_history = []

def voice_to_text(audio_path: str) -> str:
    with open(audio_path, "rb") as f:
        audio = speech.RecognitionAudio(content=f.read())

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        language_code="en-US"
    )

    response = speech_client.recognize(config=config, audio=audio)
    if not response.results:
        return "Sorry, I could not understand."
    return response.results[0].alternatives[0].transcript

def get_ai_response(text: str) -> str:
    chat_history.append({"role": "user", "parts": [text]})
    response = model.generate_content(chat_history)
    chat_history.append({"role": "model", "parts": [response.text]})
    return response.text

def text_to_voice(text: str) -> str:
    synthesis_input = texttospeech.SynthesisInput(text=text)

    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US",
        ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
    )

    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )

    response = tts_client.synthesize_speech(
        input=synthesis_input,
        voice=voice,
        audio_config=audio_config
    )

    os.makedirs("/tmp/audio", exist_ok=True)
    filename = f"/tmp/audio/{uuid.uuid4()}.mp3"

    with open(filename, "wb") as out:
        out.write(response.audio_content)

    return filename

