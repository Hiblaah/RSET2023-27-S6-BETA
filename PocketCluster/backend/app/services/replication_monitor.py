from sqlalchemy.orm import Session
from app.models.chunk_replication import ChunkReplication
from app.models.chunk import Chunk
from app.services.plan_replication import plan_replication

#If ACTIVE replicas < replication_factor → call plan_replication”

def ensure_replication(
    db: Session,
    chunk_id: int,
    replication_factor: int = 2,
):
    # Count ACTIVE replicas
    active_replicas = (
        db.query(ChunkReplication)
        .filter(
            ChunkReplication.chunk_id == chunk_id,
            ChunkReplication.replica_status == "ACTIVE"
        )
        .count()
    )

    # If less than replication factor, plan more replications
    if active_replicas < replication_factor:
        chunk = db.query(ChunkReplication).filter(ChunkReplication.chunk_id == chunk_id).first()
        if chunk:
            plan_replication(
                db,
                chunk_id=chunk_id,
                chunk_size=chunk.chunk_size,
                replicas=replication_factor
            )      
