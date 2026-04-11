package com.phonecluster.app.utils

import android.content.Context
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

data class LocalChunkItem(
    val name: String,
    val sizeBytes: Long,
    val lastModified: Long,
    val file: File
)

object LocalChunkStore {

    fun chunksDir(context: Context): File {
        val dir = File(context.filesDir, "cluster_chunks")
        if (!dir.exists()) dir.mkdirs()
        return dir
    }

    fun listChunks(context: Context): List<LocalChunkItem> {
        val dir = chunksDir(context)
        val files = dir.listFiles() ?: emptyArray()

        return files
            .filter { it.isFile }
            .sortedByDescending { it.lastModified() }
            .map {
                LocalChunkItem(
                    name = it.name,
                    sizeBytes = it.length(),
                    lastModified = it.lastModified(),
                    file = it
                )
            }
    }

    fun formatBytes(bytes: Long): String {
        return when {
            bytes < 1024 -> "$bytes B"
            bytes < 1024 * 1024 -> String.format(Locale.US, "%.2f KB", bytes / 1024.0)
            bytes < 1024 * 1024 * 1024 -> String.format(Locale.US, "%.2f MB", bytes / (1024.0 * 1024.0))
            else -> String.format(Locale.US, "%.2f GB", bytes / (1024.0 * 1024.0 * 1024.0))
        }
    }

    fun formatTime(millis: Long): String {
        val df = SimpleDateFormat("dd MMM, HH:mm:ss", Locale.US)
        return df.format(Date(millis))
    }
}
