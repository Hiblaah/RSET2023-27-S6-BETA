from app.core.database import get_db, SessionLocal
from app.core.connection_manager import manager

from app.models.chunk import Chunk
from app.models.device import Device
from app.models.file import File as FileModel   # Avoid name conflict with fastapi 'File'
from app.models.chunk import Chunk             

from app.services.chunk_waiter import set_chunk_arrived
from app.core.database import get_db

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from fastapi import Request

from sqlalchemy.orm import Session
from pathlib import Path

from app.services.file_chunk_download import fetch_all_chunks_for_file, assemble_file_from_chunks
from app.core.constants import TEMP_CHUNK_DIR

router = APIRouter()

@router.get("/files/{file_id}/download")
async def download_file(file_id: int, db: Session = Depends(get_db)):
    print("DOWNLOAD FILE ROUTE HIT:", file_id)
    try:
        # 1. Ensure all chunks are fetched locally
        chunks = await fetch_all_chunks_for_file(db, file_id, manager)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    final_path = TEMP_CHUNK_DIR / f"file_{file_id}.bin"

    try:
        # 2. Assemble chunks in order
        assemble_file_from_chunks(chunks, final_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # 3. Serve assembled file
    return FileResponse(
        final_path,
        media_type="application/octet-stream",
        filename=f"file_{file_id}.bin"
    )
@router.delete("/files/{file_id}")
async def delete_file(file_id: int, db: Session = Depends(get_db)):

    # 1. Check if file exists
    db_file = db.query(FileModel).filter(FileModel.file_id == file_id).first()
    if not db_file:
        raise HTTPException(status_code=404, detail="File not found")

    # 2. Get chunk IDs BEFORE deleting
    chunk_ids = [
        c.chunk_id
        for c in db.query(Chunk).filter(Chunk.file_id == file_id).all()
    ]

    # 3. Delete DB records
    try:
        db.delete(db_file)   # cascades to chunks + replications
        db.commit()
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Database error: {str(e)}"
        )

    # 4. Delete physical chunk files
    for chunk_id in chunk_ids:
        chunk_path = TEMP_CHUNK_DIR / f"chunk_{chunk_id}.bin"

        if chunk_path.exists():
            try:
                chunk_path.unlink()
            except Exception as e:
                print(f"Failed to delete chunk file {chunk_path}: {e}")

    # 5. Delete assembled file
    assembled_path = TEMP_CHUNK_DIR / f"file_{file_id}.bin"

    if assembled_path.exists():
        try:
            assembled_path.unlink()
        except Exception as e:
            print(f"Failed to delete assembled file {assembled_path}: {e}")

    return {
        "status": "success",
        "message": f"File {file_id} and all related data deleted."
    }