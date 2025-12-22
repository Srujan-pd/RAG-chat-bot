from fastapi import FastAPI, Request
from database import Base, engine
from auth import router as auth_router
from chat import router as chat_router
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi import FastAPI, UploadFile, File
from voice_chat import voice_to_text, get_ai_response, text_to_voice

Base.metadata.create_all(bind=engine)

# Simple startup migration: ensure `users.created_at` exists (helpful when models change during development)
from sqlalchemy import text

def ensure_users_created_at():
    try:
        with engine.begin() as conn:
            res = conn.execute(text("PRAGMA table_info('users');"))
            cols = [row[1] for row in res.fetchall()]
            if 'created_at' not in cols:
                # add column (SQLite allows adding columns)
                conn.execute(text('ALTER TABLE users ADD COLUMN created_at DATETIME;'))
                # initialize existing rows
                conn.execute(text("UPDATE users SET created_at = CURRENT_TIMESTAMP WHERE created_at IS NULL;"))
                print('Migration applied: added users.created_at column and initialized values')
            else:
                print('Migration check: users.created_at already present')
    except Exception as e:
        print('Could not ensure users.created_at column:', e)

ensure_users_created_at()

app = FastAPI(title="GenAI Chatbot")

templates = Jinja2Templates(directory="templates")


app.include_router(auth_router)
app.include_router(chat_router)

# @app.get("/", response_class=HTMLResponse)
# def home(request: Request):
#     return templates.TemplateResponse("chat.html", {"request": request})

# main.py
@app.get("/register", response_class=HTMLResponse)
def register_page(request: Request):
    return templates.TemplateResponse("register.html", {"request": request})


@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})




@app.get("/chat", response_class=HTMLResponse)
def home(request: Request):
    initial_reply = "That's a very kind question! ... How are you doing today? And what can I help you with?"
    # default demo user id is 1
    return templates.TemplateResponse(
        "chat.html",
        {"request": request, "initial_reply": initial_reply}
    )



@app.post("/voice-chat/")
async def voice_chat(file: UploadFile = File(...)):

    audio_path = f"audio/{file.filename}"

    with open(audio_path, "wb") as f:
        f.write(await file.read())

    user_text = voice_to_text(audio_path)

    bot_text = get_ai_response(user_text)

    audio_reply = text_to_voice(bot_text)

    return {
        "user_text": user_text,
        "bot_text": bot_text,
        "audio_file": audio_reply
    }