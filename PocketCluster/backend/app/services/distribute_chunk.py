import asyncio
from app.models.chunk_replication import ChunkReplication
from app.services.plan_replication import plan_replication
from app.core.constants import SERVER_BASE_URL
from app.models.chunk import Chunk

async def distribute_chunk(db, chunk_id, manager):

    # Always fetch fresh state from DB
    chunk = db.query(Chunk).filter(
        Chunk.chunk_id == chunk_id
    ).first()

    if not chunk:
        print(f"Chunk {chunk_id} no longer exists.")
        return

    # 1. Plan replication
    plan_replication(db, chunk.chunk_id, chunk.chunk_size)

    # 2. Fetch assignments
    new_assignments = db.query(ChunkReplication).filter(
        ChunkReplication.chunk_id == chunk.chunk_id,
        ChunkReplication.replica_status == "REPLICATING"
    ).all()

    if not new_assignments:
        print(f"No devices available to replicate chunk {chunk.chunk_id}")
        return

    tasks = []
    has_failed_assignments = False

    for assignment in new_assignments:

        if assignment.device_id in manager.active:

            tasks.append(
                manager.send_command(
                    device_id=assignment.device_id,
                    command_type="DOWNLOAD_CHUNK",
                    data={
                        "chunk_id": chunk.chunk_id,
                        "download_url": f"{SERVER_BASE_URL}/chunks/{chunk.chunk_id}/download",
                        "expected_hash": chunk.chunk_hash
                    }
                )
            )

            print(f"SENDING DOWNLOAD CHUNK OVER WS to {assignment.device_id}")

        else:

            print(
                f"Skipping WS send: Device {assignment.device_id} "
                f"is not connected. Marking assignment as FAILED."
            )

            assignment.replica_status = "FAILED"
            has_failed_assignments = True

    if has_failed_assignments:
        db.commit()

    if tasks:
        await asyncio.gather(*tasks)