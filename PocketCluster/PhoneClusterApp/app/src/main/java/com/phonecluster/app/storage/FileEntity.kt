package com.phonecluster.app.storage

import androidx.room.Entity
import androidx.room.PrimaryKey

@Entity(tableName = "files")
data class FileEntity(

    @PrimaryKey(autoGenerate = true)
    val localId: Int = 0,

    val serverFileId: Long,
    val fileName: String,
    val fileType: String,
    val fileDate: Long,
    val fileSize: Long,

    val embedding: FloatArray
)
