from datetime import datetime, timezone, timedelta
from sqlalchemy.orm import Session

from app.models.device import Device
from app.services.offline_utils import handle_device_offline

HEARTBEAT_TIMEOUT = timedelta(seconds=13)


def mark_offline_devices(db: Session):
    cutoff = datetime.now(timezone.utc) - HEARTBEAT_TIMEOUT

    # Find devices that timed out, regardless of current status
    devices = (
        db.query(Device)
        .filter(Device.last_heartbeat < cutoff)
        .all()
    )

    for device in devices:
        handle_device_offline(db, device.device_id)

    if devices:
        db.commit()