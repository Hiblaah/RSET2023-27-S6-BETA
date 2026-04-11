from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from pathlib import Path
from app.services.chunk_waiter import set_chunk_arrived

from app.core.database import get_db
from app.models.chunk import Chunk
from fastapi import Request
from app.core.constants import TEMP_CHUNK_DIR

router = APIRouter()


@router.get("/chunks/{chunk_id}/download")
def download_chunk(chunk_id: int, db: Session = Depends(get_db)):
    c = db.query(Chunk).filter(Chunk.chunk_id == chunk_id).first()
    if not c:
        raise HTTPException(status_code=404, detail="Chunk not found in DB")

    path = TEMP_CHUNK_DIR / f"chunk_{chunk_id}.bin"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Chunk file missing: {path}")

    return FileResponse(
        path=str(path),
        filename=f"chunk_{chunk_id}.bin",
        media_type="application/octet-stream"
    )

@router.post("/internal/ingest/{chunk_id}")
async def ingest_chunk(chunk_id: int, request: Request):
    data = await request.body()
    set_chunk_arrived(chunk_id, data)
    return {"status": "ok"}
