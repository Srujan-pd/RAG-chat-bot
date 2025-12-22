from fastapi import APIRouter, Depends, Form, HTTPException
from sqlalchemy.orm import Session
from passlib.context import CryptContext
from database import SessionLocal
from models import User

router = APIRouter(prefix="/auth")

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/register")
def register(username: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    """Register a new user. If the DB schema is missing `created_at`, attempt to patch it and retry."""
    from sqlalchemy.exc import OperationalError
    from sqlalchemy import text
    # try to find existing user; if a schema issue occurs, attempt migration and retry
    try:
        user = db.query(User).filter(User.username == username).first()
    except OperationalError as e:
        # check for missing column and try to add it
        if 'no such column' in str(e):
            try:
                from database import engine
                with engine.begin() as conn:
                    conn.execute(text('ALTER TABLE users ADD COLUMN created_at DATETIME;'))
                    conn.execute(text("UPDATE users SET created_at = CURRENT_TIMESTAMP WHERE created_at IS NULL;"))
                # retry query
                user = db.query(User).filter(User.username == username).first()
            except Exception as e2:
                raise HTTPException(status_code=500, detail=f"Migration failed: {e2}")
        else:
            raise HTTPException(status_code=500, detail=str(e))

    if user:
        raise HTTPException(status_code=400, detail="User already exists")

    hashed = pwd_context.hash(password)
    new_user = User(username=username, password=hashed)
    db.add(new_user)
    db.commit()
    return {"message": "User registered successfully"}

@router.post("/login")
def login(username: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == username).first()
    if not user or not pwd_context.verify(password, user.password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    return {"message": "Login successful", "user_id": user.id}
