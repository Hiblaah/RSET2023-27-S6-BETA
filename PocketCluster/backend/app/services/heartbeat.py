from sqlalchemy.orm import Session
from app.models.device import Device
from datetime import datetime
from typing import Optional

def handle_heartbeat(db, device_id, available_storage, mode):
    
    device = (
        db.query(Device)
        .filter(Device.device_id == device_id)
        .first()
    )

    if device is None:
        raise ValueError("Device not registered")

    device.last_heartbeat = datetime.utcnow()
    device.status = "ONLINE"
    device.mode = mode
    
    if available_storage is not None:
        if 0 <= available_storage <= device.storage_capacity:
            device.available_storage = available_storage    

    db.commit()
