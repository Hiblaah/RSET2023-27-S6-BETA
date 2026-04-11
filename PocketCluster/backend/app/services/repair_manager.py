from sqlalchemy.orm import Session
from sqlalchemy import func

from app.models.chunk import Chunk
from app.models.chunk_replication import ChunkReplication
from app.services.plan_replication import plan_replication
from app.services.distribute_chunk import distribute_chunk
from app.core.connection_manager import manager

DESIRED_REPLICAS = 2


async def repair_under_replicated_chunks(db: Session):
    chunks = db.query(Chunk).all()

    for chunk in chunks:

        active_replicas = (
            db.query(func.count(ChunkReplication.device_id))
            .filter(
                ChunkReplication.chunk_id == chunk.chunk_id,
                ChunkReplication.replica_status == "ACTIVE"
            )
            .scalar()
        )

        if active_replicas < DESIRED_REPLICAS:
            print(
                f"Repairing chunk {chunk.chunk_id}. "
                f"Active replicas={active_replicas}"
            )

            plan_replication(
                db=db,
                chunk_id=chunk.chunk_id,
                chunk_size=chunk.chunk_size,
                replicas=DESIRED_REPLICAS
            )

            await distribute_chunk(db, chunk.chunk_id, manager)