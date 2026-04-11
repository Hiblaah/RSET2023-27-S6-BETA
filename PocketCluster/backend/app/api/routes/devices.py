from sqlalchemy.orm import Session 

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File

from app.core.database import get_db
from app.core.connection_manager import manager

from app.models.chunk import Chunk
from app.models.device import Device
from app.models.file import File as FileModel   # Avoid name conflict with fastapi 'File'
from app.models.chunk import Chunk             
from sqlalchemy import func
from app.models.chunk_replication import ChunkReplication

from app.services.registration import register_device
from app.services.distribute_chunk import distribute_chunk
from app.services.heartbeat import handle_heartbeat

import hashlib
from pydantic import BaseModel
from typing import List, Optional 
from datetime import datetime, timezone

from app.core.constants import TEMP_CHUNK_DIR

router = APIRouter()

@router.post("/devices/register")
def register_device_endpoint(payload: dict, db: Session = Depends(get_db)):
    result = register_device(
        db=db,
        user_id=payload["user_id"],
        device_name=payload["device_name"],
        storage_capacity=payload["storage_capacity"],
        available_storage=payload["available_storage"],
        fingerprint=payload["fingerprint"],
    )
    return result
    

class ChunkMetadata(BaseModel):
    chunk_index: int
    chunk_hash: str
    chunk_size: int

class FileUploadInit(BaseModel):
    user_id: int
    file_name: str
    file_size: int
    num_chunks: int
    chunks: List[ChunkMetadata] # List of hashes the user calculated

class HeartbeatRequest(BaseModel):
    device_id: int
    available_storage: Optional[int] = None
    mode:str


# File upload initialization endpoint:
@router.post("/files/init")
async def initialize_upload(payload: FileUploadInit, db: Session = Depends(get_db)):
    # A. Create the File record
    new_file = FileModel(
    user_id=payload.user_id,
    file_name=payload.file_name,
    file_size=payload.file_size,
    num_chunks=payload.num_chunks,
    upload_timestamp=datetime.now(timezone.utc), 
    file_type="bin",
    )
    db.add(new_file)
    db.flush() # This generates the file_id without finishing the transaction

    # B. Create the "slots" for the Chunks
    for c in payload.chunks:
        new_chunk = Chunk(
            file_id=new_file.file_id,
            chunk_index=c.chunk_index,
            chunk_hash=c.chunk_hash,
            chunk_size=c.chunk_size
        )
        db.add(new_chunk)

    db.commit() # Save everything to DB

    return {
        "status": "success",
        "file_id": new_file.file_id,
        "message": "Metadata saved. You can now start sending chunks."
    }

@router.post("/files/{file_id}/chunks/{chunk_index}")
async def upload_chunk_data(
    file_id: int, 
    chunk_index: int,
    file: UploadFile = File(...), 
    db: Session = Depends(get_db)
):
    # 1. Find the chunk metadata we created in /init
    db_chunk = db.query(Chunk).filter(
        Chunk.file_id == file_id, 
        Chunk.chunk_index == chunk_index
    ).first()

    if not db_chunk:
        raise HTTPException(status_code=404, detail="Chunk metadata not found")

    # 2. Read the encrypted bytes
    chunk_data = await file.read()
    
    # 3. Integrity Check: Does the hash match what the phone promised in /init?
    actual_hash = hashlib.sha256(chunk_data).hexdigest()
    if actual_hash != db_chunk.chunk_hash:
        raise HTTPException(status_code=400, detail="Integrity check failed: Hash mismatch")

    # 4. Save locally temporarily
    path = TEMP_CHUNK_DIR / f"chunk_{db_chunk.chunk_id}.bin"
    with open(path, "wb") as f:
        f.write(chunk_data)

    # 5. TRIGGER REPLICATION
    # Now that this specific chunk is safe on the server, 
    # we tell the cluster to come get it.
    print("Using manager:", manager)
    await distribute_chunk(db, db_chunk.chunk_id, manager)

    return {"status": "success", "chunk_id": db_chunk.chunk_id}


# heartbeat endpoint
@router.post("/devices/heartbeat")
def heartbeat_endpoint(
    payload: HeartbeatRequest,
    db: Session = Depends(get_db)
):
    try:
        handle_heartbeat(
            db=db,
            device_id=payload.device_id,
            available_storage=payload.available_storage,
            mode = payload.mode
        )
        return {"status": "ok"}

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.get("/cluster/status")
def get_cluster_status(db: Session = Depends(get_db)):

    # ----- DEVICE STATUS -----
    devices = db.query(Device).all()

    device_list = []
    for d in devices:
        device_list.append({
            "device_id": d.device_id,
            "status": d.status,
            "mode": d.mode,
            "available_storage": d.available_storage,
            "last_heartbeat": d.last_heartbeat
        })

    # ----- FILES -----
    files = db.query(FileModel).all()

    file_list = []
    for f in files:
        file_list.append({
            "file_id": f.file_id,
            "file_name": f.file_name,
            "num_chunks": f.num_chunks,
            "file_size": f.file_size
        })

    # ----- CHUNK REPLICATION STATS -----
    chunk_stats = (
        db.query(
            ChunkReplication.replica_status,
            func.count()
        )
        .group_by(ChunkReplication.replica_status)
        .all()
    )

    replica_summary = {
        status: count for status, count in chunk_stats
    }

    # ----- TOTAL CHUNKS -----
    total_chunks = db.query(func.count(Chunk.chunk_id)).scalar()

    return {
        "devices": device_list,
        "files": file_list,
        "replica_summary": replica_summary,
        "total_chunks": total_chunks
    }