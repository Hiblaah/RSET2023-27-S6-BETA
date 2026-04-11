package com.phonecluster.app.utils

import okhttp3.MediaType.Companion.toMediaType
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.MultipartBody
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import java.security.MessageDigest
import java.util.concurrent.TimeUnit

/**
 * Metadata for a single chunk (used for /files/init request)
 */
data class ChunkMeta(
    val chunk_index: Int,
    val chunk_hash: String,
    val chunk_size: Int
)

/**
 * Handles uploading file chunks to the server
 * Uses OkHttp for network calls
 */
object ChunkUploader {

    private val JSON = "application/json".toMediaType()

    private val client = OkHttpClient.Builder()
        .connectTimeout(30, TimeUnit.SECONDS)
        .writeTimeout(10, TimeUnit.MINUTES)
        .readTimeout(10, TimeUnit.MINUTES)
        .build()

    /**
     * Compute SHA-256 hash of byte array
     */
    private fun sha256Hex(bytes: ByteArray): String {
        val md = MessageDigest.getInstance("SHA-256")
        val digest = md.digest(bytes)
        return digest.joinToString("") { "%02x".format(it) }
    }

    /**
     * Step 1: Initialize upload session
     * POST /files/init
     * Returns: file_id
     */
    fun initUpload(
        baseUrl: String,
        userId: Int,
        fileName: String,
        fileSize: Long,
        totalChunks: Int,
        chunkMetas: List<ChunkMeta>
    ): Int {
        val bodyJson = buildString {
            append("{")
            append("\"user_id\":$userId,")
            append("\"file_name\":\"${fileName.replace("\"", "\\\"")}\",")
            append("\"file_size\":$fileSize,")
            append("\"num_chunks\":$totalChunks,")
            append("\"chunks\":[")
            chunkMetas.forEachIndexed { idx, c ->
                if (idx > 0) append(",")
                append("{\"chunk_index\":${c.chunk_index},\"chunk_hash\":\"${c.chunk_hash}\",\"chunk_size\":${c.chunk_size}}")
            }
            append("]}")
        }

        val req = Request.Builder()
            .url("${baseUrl.trimEnd('/')}/files/init")
            .post(bodyJson.toRequestBody(JSON))
            .build()

        client.newCall(req).execute().use { resp ->
            val respBody = resp.body?.string().orEmpty()
            if (!resp.isSuccessful) {
                throw RuntimeException("init failed ${resp.code}: $respBody")
            }

            val match = Regex("\"file_id\"\\s*:\\s*(\\d+)").find(respBody)
                ?: throw RuntimeException("file_id not found: $respBody")
            return match.groupValues[1].toInt()
        }
    }

    /**
     * Step 2: Upload a single chunk
     * POST /files/{file_id}/chunks/{chunk_index}
     */
    fun uploadChunk(
        baseUrl: String,
        fileId: Int,
        chunk: FileChunk
    ) {
        val chunkBody = chunk.data.toRequestBody("application/octet-stream".toMediaTypeOrNull())

        val multipart = MultipartBody.Builder()
            .setType(MultipartBody.FORM)
            // MUST be "file" to match FastAPI parameter: file: UploadFile = File(...)
            .addFormDataPart("file", "chunk_${chunk.index}.bin", chunkBody)
            .build()

        val req = Request.Builder()
            .url("${baseUrl.trimEnd('/')}/files/$fileId/chunks/${chunk.index}")
            .post(multipart)
            .build()

        client.newCall(req).execute().use { resp ->
            val respBody = resp.body?.string().orEmpty()
            if (!resp.isSuccessful) {
                throw RuntimeException("chunk ${chunk.index} failed ${resp.code}: $respBody")
            }
        }
    }

    /**
     * Complete upload flow:
     * 1. Compute SHA-256 hashes for all chunks
     * 2. Initialize upload session (get file_id)
     * 3. Upload each chunk
     * 4. Return file_id
     */
    fun uploadAll(
        baseUrl: String,
        userId: Int,
        fileInfo: ChunkedFileInfo,
        chunks: List<FileChunk>,
        onProgress: (uploaded: Int, total: Int) -> Unit
    ): Int {
        // Compute SHA-256 hash for each chunk
        val metas = chunks.map { c ->
            ChunkMeta(
                chunk_index = c.index,
                chunk_hash = sha256Hex(c.data),
                chunk_size = c.size
            )
        }

        // Step 1: Initialize upload session
        val fileId = initUpload(
            baseUrl = baseUrl,
            userId = userId,
            fileName = fileInfo.name,
            fileSize = fileInfo.size,
            totalChunks = fileInfo.totalChunks,
            chunkMetas = metas
        )

        // Step 2: Upload each chunk
        chunks.forEachIndexed { i, c ->
            uploadChunk(baseUrl, fileId, c)
            onProgress(i + 1, chunks.size)
        }

        return fileId
    }
}