package com.phonecluster.app.storage

import android.content.Context
import java.io.File

object ChunkStorage {

    private const val CHUNK_DIR = "chunks"

    private fun getChunkDir(context: Context): File {
        val dir = File(context.filesDir, CHUNK_DIR)
        if (!dir.exists()) {
            dir.mkdirs()
        }
        return dir
    }

    private fun getChunkFile(context: Context, chunkId: Long): File {
        return File(getChunkDir(context), "chunk_$chunkId.bin")
    }

    fun writeChunk(context: Context, chunkId: Long, data: ByteArray) {
        val file = getChunkFile(context, chunkId)
        file.writeBytes(data)
    }

    fun readChunk(context: Context, chunkId: Long): ByteArray {
        val file = getChunkFile(context, chunkId)
        if (!file.exists()) {
            throw RuntimeException("Chunk $chunkId not found")
        }
        return file.readBytes()
    }

    fun hasChunk(context: Context, chunkId: Long): Boolean {
        return getChunkFile(context, chunkId).exists()
    }

    fun deleteChunk(context: Context, chunkId: Long) {
        getChunkFile(context, chunkId).delete()
    }
}