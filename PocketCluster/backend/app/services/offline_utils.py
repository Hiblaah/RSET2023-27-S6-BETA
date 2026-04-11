from sqlalchemy.orm import Session
from app.models.device import Device
from app.models.chunk_replication import ChunkReplication
from app.core.connection_manager import manager

def handle_device_offline(db: Session, device_id: int):
    """
    Centrally handles all logic when a device goes OFFLINE:
    1. Marks device status as OFFLINE.
    2. Marks all ACTIVE replicas on this device as LOST.
    3. Marks all REPLICATING replicas on this device as FAILED.
    4. Ensures WebSocket is disconnected.
    """
    device = db.query(Device).filter(Device.device_id == device_id).first()
    if not device:
        return

    print(f"Centrally marking device {device_id} as OFFLINE")
    device.status = "OFFLINE"

    # Update replicas
    replicas = (
        db.query(ChunkReplication)
        .filter(
            ChunkReplication.device_id == device_id,
            ChunkReplication.replica_status.in_(["ACTIVE", "REPLICATING"])
        )
        .all()
    )

    for r in replicas:
        if r.replica_status == "ACTIVE":
            print(f"Replica lost: chunk {r.chunk_id} on device {device_id}")
            r.replica_status = "LOST"
        elif r.replica_status == "REPLICATING":
            print(f"Replication failed (device offline): chunk {r.chunk_id} on device {device_id}")
            r.replica_status = "FAILED"

    # Close WebSocket connection if managed
    manager.disconnect(device_id)
