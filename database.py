import os
import time
import logging
from dotenv import load_dotenv
from sqlalchemy import create_engine, text  # Add text import
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError
from typing import Generator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Get DATABASE_URL
DATABASE_URL = os.getenv("DATABASE_URL")

# Initialize with retry logic
engine = None
SessionLocal = None
Base = declarative_base()
_initialized = False

def init_database(max_retries=5, retry_delay=3) -> bool:
    """Initialize database connection with retry logic"""
    global engine, SessionLocal, _initialized
    
    if _initialized and engine is not None:
        return True
    
    if not DATABASE_URL:
        logger.error("⚠️ DATABASE_URL not found in environment variables")
        return False
    
    logger.info(f"🔧 Initializing database connection...")
    
    for attempt in range(max_retries):
        try:
            # IMPORTANT: Do not modify the URL in any way
            engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_recycle=300)
            
            # Test the connection - Use text() wrapper
            with engine.connect() as connection:
                result = connection.execute(text("SELECT 1"))  # Fixed: added text()
                logger.info(f"   Connection test successful")
            
            SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
            _initialized = True
            logger.info("✅ Database connected successfully!")
            return True
            
        except Exception as e:
            logger.error(f"⚠️ Database connection failed (attempt {attempt + 1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                logger.info(f"🔄 Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                logger.error("❌ Database connection failed after all retries")
                engine = None
                SessionLocal = None
                _initialized = False
                return False

def get_db() -> Generator[Session, None, None]:
    """Dependency for getting database session"""
    # Try to initialize if not already done
    if not _initialized:
        logger.info("Database not initialized, attempting initialization...")
        if not init_database():
            logger.error("Database initialization failed, cannot provide session")
            # Instead of yielding None, raise an exception
            raise Exception("Database connection not available")
    
    if SessionLocal is None:
        logger.error("SessionLocal is None after initialization")
        raise Exception("Database session factory not available")
    
    db = SessionLocal()
    try:
        yield db
    except SQLAlchemyError as e:
        logger.error(f"Database session error: {e}")
        db.rollback()
        raise
    finally:
        db.close()

# Initialize on import
init_database()
