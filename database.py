import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import logging

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Get DATABASE_URL
DATABASE_URL = os.getenv("DATABASE_URL")

# Log database connection status (without exposing credentials)
if DATABASE_URL:
    masked_url = DATABASE_URL.split('@')[-1] if '@' in DATABASE_URL else DATABASE_URL
    logger.info(f"🔌 Database URL configured: ...{masked_url[:20]}")
else:
    logger.warning("⚠️ DATABASE_URL not set - running without persistence")

# IMPORTANT: Do not modify the URL in any way
# Create engine and other objects
engine = None
SessionLocal = None
Base = declarative_base()

try:
    if DATABASE_URL:
        # Direct creation without modifications
        engine = create_engine(
            DATABASE_URL,
            pool_pre_ping=True,  # Verify connections before using
            pool_size=5,
            max_overflow=10
        )
        
        # Test the connection
        with engine.connect() as connection:
            logger.info("✅ Database connected successfully!")
        
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    else:
        logger.warning("⚠️ No database URL - running in memory-only mode")
        
except Exception as e:
    logger.error(f"⚠️ Database connection failed: {e}")
    engine = None
    SessionLocal = None

def get_db():
    """Dependency for getting database session"""
    if SessionLocal is None:
        logger.warning("⚠️ Database not available - yielding None")
        yield None
        return
    
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
