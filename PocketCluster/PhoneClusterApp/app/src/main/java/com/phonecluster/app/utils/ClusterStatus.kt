package com.phonecluster.app.utils

import com.google.gson.annotations.SerializedName

data class ClusterStatusResponse(
    val devices: List<DeviceInfo>,
    val files: List<FileInfo>,

    @SerializedName("replica_summary")
    val replicaSummary: Map<String, Int>,

    @SerializedName("total_chunks")
    val totalChunks: Int
)

data class DeviceInfo(
    @SerializedName("device_id")
    val deviceId: Int,

    val status: String,
    val mode: String,

    @SerializedName("available_storage")
    val availableStorage: Long,

    @SerializedName("last_heartbeat")
    val lastHeartbeat: String
)

data class FileInfo(
    @SerializedName("file_id")
    val fileId: Int,

    @SerializedName("file_name")
    val fileName: String,

    @SerializedName("num_chunks")
    val numChunks: Int,

    @SerializedName("file_size")
    val fileSize: Long
)