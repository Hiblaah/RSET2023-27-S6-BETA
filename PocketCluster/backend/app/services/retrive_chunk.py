from app.models.chunk_replication import ChunkReplication
from app.models.device import Device
import hashlib
from app.core.constants import SERVER_BASE_URL
from pathlib import Path
from app.core.constants import TEMP_CHUNK_DIR
from app.services.chunk_waiter import init_chunk_wait, wait_for_chunk


async def retrieve_chunk(db, chunk, manager):
    replicas = (
        db.query(ChunkReplication)
        .filter(
            ChunkReplication.chunk_id == chunk.chunk_id,
            ChunkReplication.replica_status == "ACTIVE"
        )
        .all()
    )

    if not replicas:
        raise RuntimeError("No active replicas")

    for replica in replicas:
        device = (
            db.query(Device)
            .filter(
                Device.device_id == replica.device_id,
                Device.status == "ONLINE"
            )
            .first()
        )

        if not device:
            continue

        init_chunk_wait(chunk.chunk_id)

        try:
            await manager.send_command(
                device.device_id,
                "PUSH_CHUNK",
                {
                    "chunk_id": chunk.chunk_id,
                    "target_url": f"{SERVER_BASE_URL}/internal/ingest/{chunk.chunk_id}"
                }
            )

            data = await wait_for_chunk(chunk.chunk_id)

            # integrity check
            if hashlib.sha256(data).hexdigest() != chunk.chunk_hash:
                raise ValueError("Hash mismatch")
    
            temp_path = TEMP_CHUNK_DIR / f"chunk_{chunk.chunk_id}.bin"
            temp_path.parent.mkdir(parents=True, exist_ok=True)

            with open(temp_path, "wb") as f:
                f.write(data)

            return temp_path  # return file path, not raw bytes

        except Exception:
            continue

    raise RuntimeError(f"Chunk {chunk.chunk_id} unrecoverable")
