import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Get DATABASE_URL
DATABASE_URL = os.getenv("DATABASE_URL")

# Log (without exposing full URL for security)
if DATABASE_URL:
    logger.info("✅ DATABASE_URL is configured")
    # Show only first part for debugging
    if "postgresql://" in DATABASE_URL:
        logger.info("📊 Database type: PostgreSQL")
else:
    logger.warning("⚠️ DATABASE_URL not found in environment variables")
    # Try to create a fallback for testing
    DATABASE_URL = "sqlite:///./test.db"
    logger.warning(f"⚠️ Using fallback SQLite database: {DATABASE_URL}")

# Create engine and other objects
engine = None
SessionLocal = None
Base = declarative_base()

try:
    # Create engine with retry settings for Cloud Run
    engine = create_engine(
        DATABASE_URL,
        pool_pre_ping=True,  # Verify connections before using
        pool_recycle=3600,   # Recycle connections every hour
        echo=False           # Set to True for SQL debugging
    )
    
    # Test connection
    with engine.connect() as conn:
        logger.info("✅ Database connected successfully!")
    
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
except Exception as e:
    logger.error(f"❌ Database connection failed: {e}")
    import traceback
    traceback.print_exc()
    # Don't crash - allow app to start without database
    engine = None
    SessionLocal = None

def get_db():
    """Dependency to get DB session"""
    if SessionLocal is None:
        logger.error("❌ Database not available")
        # Yield a mock session or handle gracefully
        yield None
        return
        
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
