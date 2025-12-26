from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from database import Base, engine
from auth import router as auth_router
from chat import router as chat_router
from fastapi.templating import Jinja2Templates
import models

# This creates your tables automatically
Base.metadata.create_all(bind=engine)

app = FastAPI(title="GenAI Chatbot")
templates = Jinja2Templates(directory="templates")

app.include_router(auth_router)
app.include_router(chat_router)

@app.get("/", response_class=HTMLResponse)
def root():
    return RedirectResponse(url="/login")

@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.get("/register", response_class=HTMLResponse)
def register_page(request: Request):
    return templates.TemplateResponse("register.html", {"request": request})

@app.get("/chat-ui", response_class=HTMLResponse)
def chat_page(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})
