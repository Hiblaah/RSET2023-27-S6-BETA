from sqlalchemy import Column, Integer, String, Enum, DateTime, ForeignKey
from sqlalchemy.sql import func
from app.core.database import Base
    
class Device(Base):
    __tablename__ = "devices"

    device_id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.user_id"), nullable=False)

    device_fingerprint = Column(String(255), unique=True, nullable=False, index=True)
    device_name = Column(String(255), nullable=False)

    storage_capacity = Column(Integer, nullable=False)
    available_storage = Column(Integer, nullable=False)

    status = Column(Enum("OFFLINE", "ONLINE"), default="OFFLINE", nullable=False)
    last_heartbeat = Column(DateTime, nullable=True)

    created_at = Column(DateTime, server_default=func.now())

    mode = Column(Enum("User", "Cluster"), nullable=False, default="User")
