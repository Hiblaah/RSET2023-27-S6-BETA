from sqlalchemy import Column, Integer, String, BigInteger, TIMESTAMP, ForeignKey
from app.core.database import Base

class File(Base):
    __tablename__ = "files"

    file_id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.user_id"), nullable=False)

    file_name = Column(String(255), nullable=False)
    file_size = Column(BigInteger, nullable=False)
    file_type = Column(String(100))

    upload_timestamp = Column(TIMESTAMP, nullable=False)
    num_chunks = Column(Integer, nullable=False)
