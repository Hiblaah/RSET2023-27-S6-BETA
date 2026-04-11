import os
import hashlib
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.chunk import Chunk

from app.core.constants import TEMP_CHUNK_DIR
from app.services.retrive_chunk import retrieve_chunk

def chunk_path(chunk_id: int) -> str:
    return os.path.join(TEMP_CHUNK_DIR, f"chunk_{chunk_id}.bin")

def assemble_file_from_chunks(chunks, output_path: str):
    with open(output_path, "wb") as out:
        for chunk in chunks:
            path = chunk_path(chunk.chunk_id)

            if not os.path.exists(path):
                raise RuntimeError(f"Missing chunk {chunk.chunk_id}")

            with open(path, "rb") as f:
                out.write(f.read())

async def fetch_all_chunks_for_file(db: Session, file_id: int, manager):
    chunks = (
        db.query(Chunk)
        .filter(Chunk.file_id == file_id)
        .order_by(Chunk.chunk_index)
        .all()
    )

    if not chunks:
        raise RuntimeError("No chunks found for file")

    os.makedirs(TEMP_CHUNK_DIR, exist_ok=True)

    for chunk in chunks:
        path = chunk_path(chunk.chunk_id)

        # Cache hit
        if os.path.exists(path):
            continue

        # Orchestrate retrieval from phones
        await retrieve_chunk(db, chunk, manager)

        
    return chunks
