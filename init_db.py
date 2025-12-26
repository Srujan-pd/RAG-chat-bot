import sys
import os

# ensure current directory is in PYTHONPATH
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database import Base, engine
from models import User, Chat

def init_db():
    Base.metadata.create_all(bind=engine)
    print("✅ Database initialized successfully")

if __name__ == "__main__":
    init_db()

