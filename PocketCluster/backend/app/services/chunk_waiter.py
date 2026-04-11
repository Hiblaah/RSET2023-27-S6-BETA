import asyncio

chunk_events: dict[str, asyncio.Event] = {}
chunk_data: dict[str, bytes] = {}

def init_chunk_wait(chunk_id: str):
    ev = asyncio.Event()
    chunk_events[chunk_id] = ev
    return ev

def set_chunk_arrived(chunk_id: str, data: bytes):
    chunk_data[chunk_id] = data
    chunk_events[chunk_id].set()

async def wait_for_chunk(chunk_id: str, timeout: int = 30) -> bytes:
    ev = chunk_events[chunk_id]
    await asyncio.wait_for(ev.wait(), timeout=timeout)
    return chunk_data.pop(chunk_id)
