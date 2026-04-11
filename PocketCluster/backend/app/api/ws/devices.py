from fastapi import WebSocket, WebSocketDisconnect, status
from datetime import datetime
import logging

from app.models.device import Device
from app.models.chunk_replication import ChunkReplication
from app.core.database import SessionLocal
from app.core.connection_manager import manager

logger = logging.getLogger("uvicorn")


async def device_ws(ws: WebSocket):
    logger.info("WS: Incoming connection")
    await ws.accept()
    logger.info("WS: Accepted connection")

    current_device_id = None

    try:
        logger.info("WS: Waiting for register payload...")
        payload = await ws.receive_json()
        logger.info(f"WS: Received payload: {payload}")

        if payload.get("type") != "register":
            logger.warning("WS: First message was not 'register'. Closing.")
            await ws.close(code=status.WS_1008_POLICY_VIOLATION)
            return

        fingerprint = payload.get("fingerprint")
        logger.info(f"WS: Fingerprint received: {fingerprint}")

        # ---- DB session only for registration ----
        db = SessionLocal()
        try:
            device = db.query(Device).filter(
                Device.device_fingerprint == fingerprint
            ).first()

            if not device:
                logger.warning("WS: Device not found in DB. Closing.")
                await ws.close(
                    code=status.WS_1008_POLICY_VIOLATION,
                    reason="Device not registered"
                )
                return

            logger.info(
                f"WS: Device found. ID={device.device_id}, Status={device.status}"
            )

            if device.status != "ONLINE":
                logger.info(
                    f"WS: Device {device.device_id} transitioning to ONLINE"
                )
                device.status = "ONLINE"
                device.last_seen = datetime.utcnow()
                db.commit()

            current_device_id = device.device_id

        finally:
            db.close()

        # ---- register websocket ----
        logger.info(
            f"WS: Registering device {current_device_id} with connection manager"
        )
        await manager.connect(current_device_id, ws)

        logger.info(f"WS: Sending ready to device {current_device_id}")
        await ws.send_json({
            "type": "ready",
            "device_id": current_device_id,
        })

        logger.info("WS: Entering main loop")

        # ---- main websocket loop ----
        while True:
            logger.info(f"WS: Waiting for message from {current_device_id}")
            msg = await ws.receive_json()
            logger.info(
                f"WS: Message received from {current_device_id}: {msg}"
            )

            msg_type = msg.get("type")

            if msg_type == "CHUNK_STORED_SUCCESS":
                chunk_id = msg.get("chunk_id")

                logger.info(
                    f"WS: CHUNK_STORED_SUCCESS received. "
                    f"Device={current_device_id}, Chunk={chunk_id}"
                )

                db = SessionLocal()
                try:
                    replication_entry = db.query(ChunkReplication).filter(
                        ChunkReplication.chunk_id == chunk_id,
                        ChunkReplication.device_id == current_device_id
                    ).first()

                    if replication_entry:
                        logger.info(
                            "WS: Replication entry found. Marking ACTIVE."
                        )
                        replication_entry.replica_status = "ACTIVE"
                        db.commit()

                        logger.info(
                            f"WS: Chunk {chunk_id} ACTIVE on "
                            f"Device {current_device_id}"
                        )
                    else:
                        logger.warning(
                            f"WS: No replication entry found for "
                            f"Device={current_device_id}, Chunk={chunk_id}"
                        )
                finally:
                    db.close()

            elif msg_type == "CHUNK_STORE_FAILED":
                chunk_id = msg.get("chunk_id")

                logger.warning(
                    f"WS: Chunk storage failed on device "
                    f"{current_device_id}, chunk={chunk_id}"
                )

                # optional: mark replication failed
                db = SessionLocal()
                try:
                    entry = db.query(ChunkReplication).filter(
                        ChunkReplication.chunk_id == chunk_id,
                        ChunkReplication.device_id == current_device_id
                    ).first()

                    if entry:
                        entry.replica_status = "FAILED"
                        db.commit()
                finally:
                    db.close()

            else:
                logger.warning(
                    f"WS: Unknown message type from "
                    f"{current_device_id}: {msg_type}"
                )

    except WebSocketDisconnect:
        logger.info(f"WS: Device {current_device_id} disconnected.")

    except Exception as e:
        logger.exception(
            f"WS: Unexpected error for device {current_device_id}: {e}"
        )

    finally:
        logger.info(f"WS: Cleaning up connection for {current_device_id}")

        if current_device_id:
            # Mark device as OFFLINE and all its replicas as LOST centrally
            db = SessionLocal()
            try:
                from app.services.offline_utils import handle_device_offline
                handle_device_offline(db, current_device_id)
                db.commit()
            finally:
                db.close()
            logger.info(f"WS: Disconnected {current_device_id} and marked replicas as LOST")