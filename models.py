from sqlalchemy import Column, Integer, String, Text, DateTime
from sqlalchemy.sql import func
from database import Base

class Chat(Base):
    __tablename__ = "chats_info"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(100), index=True)      
    session_id = Column(String(100), index=True)   
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
