import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Load environment variables
load_dotenv()

# Get DATABASE_URL
DATABASE_URL = os.getenv("DATABASE_URL")

print(f"👉 DATABASE_URL = {DATABASE_URL}")
print(f"👉 URL repr = {repr(DATABASE_URL)}")
print(f"👉 '..' in URL? {'..' in DATABASE_URL}")

# IMPORTANT: Do not modify the URL in any way
# Create engine and other objects
engine = None
SessionLocal = None
Base = declarative_base()

try:
    # Direct creation without modifications
    engine = create_engine(DATABASE_URL)
    
    # Test the connection
    with engine.connect() as connection:
        print("✅ Database connected successfully!")
    
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
except Exception as e:
    print(f"⚠️ Database connection failed: {e}")
    import traceback
    traceback.print_exc()
    engine = None
    SessionLocal = None

def get_db():
    if SessionLocal is None:
        yield None
        return
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
