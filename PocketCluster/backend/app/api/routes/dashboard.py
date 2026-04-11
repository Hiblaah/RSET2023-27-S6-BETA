from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.core.database import get_db
from app.models.user import User
from app.models.device import Device
from app.models.file import File as FileModel
from app.models.chunk import Chunk
from app.models.chunk_replication import ChunkReplication

router = APIRouter()


@router.get("/admin/dashboard")
def admin_dashboard(db: Session = Depends(get_db)):

    # ── Overview counts ──────────────────────────────────────────
    total_users = db.query(func.count(User.user_id)).scalar() or 0
    total_devices = db.query(func.count(Device.device_id)).scalar() or 0
    total_files = db.query(func.count(FileModel.file_id)).scalar() or 0
    total_chunks = db.query(func.count(Chunk.chunk_id)).scalar() or 0

    total_storage_used = db.query(func.coalesce(func.sum(FileModel.file_size), 0)).scalar()

    cluster_online_devices = db.query(Device).filter(
        Device.mode == "Cluster",
        Device.status == "ONLINE"
    )

    total_capacity = cluster_online_devices.with_entities(
        func.coalesce(func.sum(Device.storage_capacity), 0)
    ).scalar()

    total_available = cluster_online_devices.with_entities(
        func.coalesce(func.sum(Device.available_storage), 0)
    ).scalar()

    online_devices = (
        db.query(func.count(Device.device_id))
        .filter(Device.status == "ONLINE")
        .scalar() or 0
    )
    offline_devices = total_devices - online_devices

    cluster_devices = (
        db.query(func.count(Device.device_id))
        .filter(Device.mode == "Cluster")
        .scalar() or 0
    )

    # ── Users ────────────────────────────────────────────────────
    users = db.query(User).all()
    user_file_counts = dict(
        db.query(FileModel.user_id, func.count(FileModel.file_id))
        .group_by(FileModel.user_id)
        .all()
    )
    user_list = []
    for u in users:
        user_list.append({
            "user_id": u.user_id,
            "email": u.email,
            "created_at": str(u.created_at) if u.created_at else None,
            "last_login": str(u.last_login) if u.last_login else None,
            "file_count": user_file_counts.get(u.user_id, 0),
        })

    # ── Devices ──────────────────────────────────────────────────
    devices = db.query(Device).all()
    device_list = []
    for d in devices:
        device_list.append({
            "device_id": d.device_id,
            "user_id": d.user_id,
            "device_name": d.device_name,
            "status": d.status,
            "mode": d.mode,
            "storage_capacity": d.storage_capacity,
            "available_storage": d.available_storage,
            "last_heartbeat": str(d.last_heartbeat) if d.last_heartbeat else None,
            "created_at": str(d.created_at) if d.created_at else None,
        })

    # ── Files ────────────────────────────────────────────────────
    files = db.query(FileModel).all()
    # Build user-email lookup
    user_emails = {u.user_id: u.email for u in users}
    file_list = []
    for f in files:
        file_list.append({
            "file_id": f.file_id,
            "user_id": f.user_id,
            "owner_email": user_emails.get(f.user_id, "Unknown"),
            "file_name": f.file_name,
            "file_size": f.file_size,
            "file_type": f.file_type,
            "upload_timestamp": str(f.upload_timestamp) if f.upload_timestamp else None,
            "num_chunks": f.num_chunks,
        })

    # ── Replication summary ──────────────────────────────────────
    chunk_stats = (
        db.query(ChunkReplication.replica_status, func.count())
        .group_by(ChunkReplication.replica_status)
        .all()
    )
    replica_summary = {status: count for status, count in chunk_stats}

    # ── Chunk details (with replication info) ────────────────────
    chunks = db.query(Chunk).all()
    chunk_list = []
    for c in chunks:
        replications = (
            db.query(ChunkReplication)
            .filter(ChunkReplication.chunk_id == c.chunk_id)
            .all()
        )
        chunk_list.append({
            "chunk_id": c.chunk_id,
            "file_id": c.file_id,
            "chunk_index": c.chunk_index,
            "chunk_size": c.chunk_size,
            "chunk_hash": c.chunk_hash,
            "replicas": [
                {
                    "device_id": r.device_id,
                    "status": r.replica_status,
                }
                for r in replications
            ],
        })

    return {
        "overview": {
            "total_users": total_users,
            "total_devices": total_devices,
            "total_files": total_files,
            "total_chunks": total_chunks,
            "total_storage_used": total_storage_used,
            "total_capacity": total_capacity,
            "total_available": total_available,
            "online_devices": online_devices,
            "offline_devices": offline_devices,
            "cluster_devices": cluster_devices,
        },
        "users": user_list,
        "devices": device_list,
        "files": file_list,
        "chunks": chunk_list,
        "replica_summary": replica_summary,
    }
