import json
import os
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# ---------------------------
# Global state
# ---------------------------
KB_READY = False
KNOWLEDGE_DATA = []

DATA_PATH = "data/scraped_data.json"

print("🔥 chat.py loaded")

# ---------------------------
# Load Knowledge Base (RUNS ON CONTAINER START)
# ---------------------------
def load_knowledge_base():
    global KB_READY, KNOWLEDGE_DATA

    print("🔥 Starting knowledge base load")

    if not os.path.exists(DATA_PATH):
        print(f"❌ KB file not found at {DATA_PATH}")
        return

    try:
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            KNOWLEDGE_DATA = json.load(f)

        KB_READY = True
        print(f"✅ KB loaded successfully | Pages: {len(KNOWLEDGE_DATA)}")

    except Exception as e:
        print(f"❌ Failed to load KB: {e}")


# 🔥 THIS IS THE KEY LINE (runs in Cloud Run)
load_knowledge_base()

# ---------------------------
# Request / Response models
# ---------------------------
class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


# ---------------------------
# Chat endpoint
# ---------------------------
@app.post("/chat/")
def chat(req: ChatRequest):

    if not KB_READY:
        return {
            "status": "success",
            "message": "I'm still loading the knowledge base. Please try again in a moment.",
            "session_id": req.session_id,
        }

    user_question = req.message.lower()

    # VERY SIMPLE retrieval (replace later with embeddings)
    for page in KNOWLEDGE_DATA:
        if user_question in page.get("content", "").lower():
            return {
                "status": "success",
                "message": page["content"][:800],
                "session_id": req.session_id,
            }

    return {
        "status": "success",
        "message": "I couldn’t find that on Primis Digital yet. Can you rephrase?",
        "session_id": req.session_id,
    }

