from sqlalchemy import Column, Integer, Enum, ForeignKey, UniqueConstraint
from app.core.database import Base

class ChunkReplication(Base):
    __tablename__ = "chunk_replication"

    replication_id = Column(Integer, primary_key=True, index=True)

    chunk_id = Column(
        Integer,
        ForeignKey("chunks.chunk_id", ondelete="CASCADE"),
        nullable=False,
    )

    device_id = Column(
        Integer,
        ForeignKey("devices.device_id", ondelete="CASCADE"),
        nullable=False,
    )

    replica_status = Column(
        Enum("REPLICATING", "ACTIVE", "LOST", "FAILED", name="replica_status_enum"),
        nullable=False,
        default="REPLICATING",
    )

    __table_args__ = (
        UniqueConstraint("chunk_id", "device_id", name="uq_chunk_device"),
    )
