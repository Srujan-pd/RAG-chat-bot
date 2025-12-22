import google.generativeai as genai
import speech_recognition as sr
from gtts import gTTS
import uuid
import os

genai.configure(api_key="AIzaSyBsliImaSewrZ8ENliMuvfy_FH7jfx6PeQ")

model = genai.GenerativeModel("models/gemini-flash-latest")

chat_history = []


def voice_to_text(audio_file_path):
    recognizer = sr.Recognizer()

    with sr.AudioFile(audio_file_path) as source:
        audio = recognizer.record(source)

    try:
        text = recognizer.recognize_google(audio)
        return text
    except:
        return "Sorry, I could not understand."


def get_ai_response(user_text):

    chat_history.append({
        "role": "user",
        "parts": [user_text]
    })

    response = model.generate_content(chat_history)

    bot_text = response.text

    chat_history.append({
        "role": "model",
        "parts": [bot_text]
    })

    return bot_text


def text_to_voice(text):
    filename = f"audio/{uuid.uuid4()}.mp3"

    tts = gTTS(text=text, lang="en")
    tts.save(filename)

    return filename
