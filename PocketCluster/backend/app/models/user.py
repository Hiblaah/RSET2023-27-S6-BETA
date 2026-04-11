from sqlalchemy import Column, Integer, String, TIMESTAMP
from app.core.database import Base

class User(Base):
    __tablename__ = "users"

    user_id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), nullable=False, unique=True)
    password_hash = Column(String(255), nullable=False)

    created_at = Column(TIMESTAMP, nullable=False)
    last_login = Column(TIMESTAMP, nullable=True)
