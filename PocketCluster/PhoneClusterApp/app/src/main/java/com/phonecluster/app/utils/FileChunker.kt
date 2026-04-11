package com.phonecluster.app.utils

import android.content.Context
import android.net.Uri

data class FileChunk(
    val index: Int,
    val data: ByteArray,
    val size: Int
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false
        other as FileChunk
        return index == other.index && size == other.size
    }

    override fun hashCode(): Int {
        var result = index
        result = 31 * result + size
        return result
    }
}

data class ChunkedFileInfo(
    val name: String,
    val size: Long,
    val mimeType: String?,
    val totalChunks: Int
)

object FileChunker {

    private const val CHUNK_SIZE = 40 * 1024
    private const val ENCRYPTION_PASSWORD = "mypassword123"

    fun getFileInfo(context: Context, fileUri: Uri): ChunkedFileInfo? {
        return try {
            val contentResolver = context.contentResolver
            val cursor = contentResolver.query(fileUri, null, null, null, null)

            cursor?.use {
                if (it.moveToFirst()) {

                    val nameIndex = it.getColumnIndex(android.provider.OpenableColumns.DISPLAY_NAME)
                    val sizeIndex = it.getColumnIndex(android.provider.OpenableColumns.SIZE)

                    val name = if (nameIndex != -1) it.getString(nameIndex) else "unknown"
                    val size = if (sizeIndex != -1) it.getLong(sizeIndex) else 0L
                    val mimeType = contentResolver.getType(fileUri)

                    val totalChunks = ((size + CHUNK_SIZE - 1) / CHUNK_SIZE).toInt()

                    ChunkedFileInfo(name, size, mimeType, totalChunks)
                } else {
                    null
                }
            }

        } catch (e: Exception) {
            e.printStackTrace()
            null
        }
    }

    fun chunkFile(
        context: Context,
        fileUri: Uri,
        onProgress: (Int, Int) -> Unit = { _, _ -> }
    ): List<FileChunk> {

        val chunks = mutableListOf<FileChunk>()

        try {

            val originalBytes = FileHelper.readAllBytes(context, fileUri)

            // Encrypt WITHOUT AAD
            val encryptedBytes = AsconFileCrypto.encryptFile(
                plaintext = originalBytes,
                password = ENCRYPTION_PASSWORD,
                aad = byteArrayOf()
            )

            val totalChunks = (encryptedBytes.size + CHUNK_SIZE - 1) / CHUNK_SIZE

            var chunkIndex = 0
            var offset = 0

            while (offset < encryptedBytes.size) {

                val end = minOf(offset + CHUNK_SIZE, encryptedBytes.size)
                val chunkData = encryptedBytes.copyOfRange(offset, end)

                chunks.add(
                    FileChunk(
                        index = chunkIndex,
                        data = chunkData,
                        size = chunkData.size
                    )
                )

                chunkIndex++
                offset = end

                onProgress(chunkIndex, totalChunks)
            }

        } catch (e: Exception) {
            e.printStackTrace()
        }

        return chunks
    }

    fun formatFileSize(bytes: Long): String {
        return when {
            bytes < 1024 -> "$bytes B"
            bytes < 1024 * 1024 -> String.format("%.2f KB", bytes / 1024.0)
            bytes < 1024 * 1024 * 1024 -> String.format("%.2f MB", bytes / (1024.0 * 1024.0))
            else -> String.format("%.2f GB", bytes / (1024.0 * 1024.0 * 1024.0))
        }
    }
}