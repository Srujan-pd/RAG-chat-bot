import os
import logging
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get DATABASE_URL
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    logger.error("DATABASE_URL environment variable is not set!")
    raise ValueError("DATABASE_URL environment variable is not set!")

logger.info(f"Database URL configured: {DATABASE_URL[:30]}...")

# Create engine with connection pooling
try:
    engine = create_engine(
        DATABASE_URL,
        pool_size=5,
        max_overflow=10,
        pool_timeout=30,
        pool_recycle=1800,  # Recycle connections every 30 minutes
        echo=False
    )
    
    # Test connection
    with engine.connect() as connection:
        logger.info("✅ Database connected successfully!")
    
    SessionLocal = sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=engine
    )
    
    Base = declarative_base()
    
except Exception as e:
    logger.error(f"❌ Database connection failed: {e}")
    # Create fallback objects to prevent crashes
    engine = None
    SessionLocal = None
    Base = None


def get_db():
    """Dependency to get database session"""
    if SessionLocal is None:
        logger.error("Database session not initialized")
        yield None
        return
        
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        logger.error(f"Database session error: {e}")
        db.rollback()
        raise
    finally:
        db.close()
