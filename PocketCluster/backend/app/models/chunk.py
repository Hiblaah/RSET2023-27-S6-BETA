from sqlalchemy import Column, Integer, String, ForeignKey
from app.core.database import Base

class Chunk(Base):
    __tablename__ = "chunks"

    chunk_id = Column(Integer, primary_key=True, index=True)
    file_id = Column(Integer, ForeignKey("files.file_id"), nullable=False)

    chunk_index = Column(Integer, nullable=False)
    chunk_hash = Column(String(255), nullable=False)
    chunk_size = Column(Integer, nullable=False)
