import os
import time
import logging
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError

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

def init_database(max_retries=5, retry_delay=3):
    """Initialize database connection with retry logic"""
    global engine, SessionLocal
    
    if engine is not None:
        return engine
    
    if not DATABASE_URL:
        logger.error("⚠️ DATABASE_URL not found in environment variables")
        logger.error("   Current environment variables:")
        for key in os.environ:
            if 'DATABASE' in key.upper() or 'POSTGRES' in key.upper():
                logger.error(f"   {key}: {'SET' if os.getenv(key) else 'NOT SET'}")
        return None
    
    logger.info(f"🔧 Initializing database connection...")
    logger.info(f"   DATABASE_URL preview: {DATABASE_URL[:50]}...")
    
    for attempt in range(max_retries):
        try:
            # IMPORTANT: Do not modify the URL in any way
            engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_recycle=300)
            
            # Test the connection
            with engine.connect() as connection:
                result = connection.execute(text("SELECT 1"))
                logger.info(f"   Connection test: {result.fetchone()}")
            
            SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
            logger.info("✅ Database connected successfully!")
            return engine
            
        except Exception as e:
            logger.error(f"⚠️ Database connection failed (attempt {attempt + 1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                logger.info(f"🔄 Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                logger.error("❌ Database connection failed after all retries")
                engine = None
                SessionLocal = None
                return None

# Initialize on import (optional, can be called explicitly)
# init_database()

def get_db():
    """Dependency for getting database session"""
    if SessionLocal is None:
        # Try to initialize if not already done
        logger.info("Database session requested but not initialized, attempting initialization...")
        init_database()
        
    if SessionLocal is None:
        logger.error("Database connection not available after initialization attempt")
        raise Exception("Database connection not available")
        
    db = SessionLocal()
    try:
        yield db
    except SQLAlchemyError as e:
        logger.error(f"Database session error: {e}")
        db.rollback()
        raise
    finally:
        db.close()
