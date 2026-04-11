import asyncio
from app.core.database import SessionLocal
from app.services.offline_detection import mark_offline_devices
import app.models  # register models
from app.services.repair_manager import repair_under_replicated_chunks

async def offline_monitor_loop():
    while True:
        await asyncio.sleep(30)

        db = SessionLocal()
        try:
            mark_offline_devices(db)
        except Exception as e:
            print("Offline detection error:", e)
            db.rollback()
        finally:
            db.close()



async def repair_loop():
    while True:
        db = SessionLocal()

        try:
            await repair_under_replicated_chunks(db)
        finally:
            db.close()

        await asyncio.sleep(10)