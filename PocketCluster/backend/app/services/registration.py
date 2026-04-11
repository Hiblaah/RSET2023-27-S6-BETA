from sqlalchemy.orm import Session
from app.models.device import Device
from datetime import datetime

def register_device(
    db: Session,
    user_id: int,
    device_name: str,
    storage_capacity: int,
    available_storage: int,
    fingerprint: str,
):
    print(">>> register_device CALLED <<<")
    print("fingerprint:", fingerprint)
    # Check if device already exists
    existing = (
        db.query(Device)
        .filter(Device.device_fingerprint == fingerprint)
        .first()
    )

    if existing:
        return {
            "device_id": existing.device_id,
            "status": "already_registered"
        }

    # Otherwise create new device
    device = Device(
        user_id = user_id,
        device_name = device_name,
        device_fingerprint = fingerprint,
        status = "OFFLINE",
        last_heartbeat = datetime.utcnow(),
        storage_capacity = storage_capacity,
        available_storage = available_storage,
        created_at = datetime.utcnow(),
        mode = "User"
    )

    db.add(device)
    db.commit()
    print(">>> commit done <<<")
    db.refresh(device)

    return {
        "device_id": device.device_id,
        "status": "registered"
    }
