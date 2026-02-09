import os
import logging
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    logger.error("DATABASE_URL environment variable is not set!")
    engine = None
    SessionLocal = None
    Base = None
else:
    try:
        logger.info("Initializing database connection...")
        
        engine = create_engine(
            DATABASE_URL,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,
            pool_recycle=300,
            echo=False
        )
        
        # Test connection
        with engine.connect() as connection:
            logger.info("✅ Database connected successfully")
        
        SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=engine
        )
        
        Base = declarative_base()
        
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
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
