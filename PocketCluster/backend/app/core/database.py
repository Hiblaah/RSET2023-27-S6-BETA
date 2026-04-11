from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

DATABASE_URL = (
    "mysql+mysqlconnector://storage_user:1234@localhost/phone_storage"
)

engine = create_engine(
    DATABASE_URL,
    pool_size=100,
    max_overflow=40,
    pool_timeout=30,
    pool_recycle=1800
)

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False
)
Base = declarative_base()

from app.models.user import User
#from app.models.device import Device