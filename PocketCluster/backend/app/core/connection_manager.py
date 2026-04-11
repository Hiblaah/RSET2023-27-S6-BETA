from typing import Dict
from fastapi import WebSocket
import logging 

logger = logging.getLogger("uvicorn")

class ConnectionManager:
    def __init__(self):
        self.active: Dict[int, WebSocket] = {}

    async def connect(self, device_id: int, websocket: WebSocket):
        self.active[device_id] = websocket

    def disconnect(self, device_id: int):
        self.active.pop(device_id, None)

    async def send_json_to_device(self, device_id: int, payload: dict) -> bool:
        ws = self.active.get(device_id)

        if not ws:
            logger.error(
                f"WS SEND FAILED: Device {device_id} not connected. "
                f"Active devices: {list(self.active.keys())}"
            )
            return False

        try:
            await ws.send_json(payload)
            logger.info(f"WS SEND SUCCESS to device {device_id}")
            return True
        except Exception as e:
            logger.exception(f"WS SEND ERROR to device {device_id}: {e}")
            return False
    
    async def send_command(self, device_id: int, command_type: str, data: dict) -> bool:
        payload = {
            "type": "command",
            "command": command_type,
            "data": data
        }
        return await self.send_json_to_device(device_id, payload)

# ✅ THIS is what your import expects
manager = ConnectionManager()