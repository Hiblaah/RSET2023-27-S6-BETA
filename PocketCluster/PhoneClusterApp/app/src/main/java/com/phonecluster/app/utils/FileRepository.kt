package com.phonecluster.app.utils

import com.phonecluster.app.core.SERVER_BASE_URL
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.OkHttpClient
import okhttp3.Request
import java.io.File
import android.content.Context
import android.util.Log
class FileRepository {

    private val client = OkHttpClient()

    companion object {
        private const val ENCRYPTION_PASSWORD = "mypassword123"
    }

    suspend fun downloadFile(
        context: Context,
        fileId: Long,
        fileName: String
    ) = withContext(Dispatchers.IO) {

        val request = Request.Builder()
            .url("$SERVER_BASE_URL/files/$fileId/download")
            .get()
            .build()

        val response = client.newCall(request).execute()

        if (!response.isSuccessful) {
            throw RuntimeException("Download failed: ${response.code}")
        }

        val encryptedBytes = response.body?.bytes()
            ?: throw RuntimeException("Empty response")

        val decryptedBytes = AsconFileCrypto.decryptFile(
            encryptedFile = encryptedBytes,
            password = ENCRYPTION_PASSWORD,
            aad = byteArrayOf()
        )

        saveFileLocally(context, fileName, decryptedBytes)
    }

    private fun saveFileLocally(context: Context, fileName: String, data: ByteArray) {

        val downloadsDir = File("/storage/emulated/0/Download")

        if (!downloadsDir.exists()) {
            downloadsDir.mkdirs()
        }

        val file = File(downloadsDir, fileName)

        file.writeBytes(data)

        Log.d("DOWNLOAD_DEBUG", "File saved: ${file.absolutePath}")

        DownloadNotificationHelper.showDownloadComplete(context, file)
    }

    suspend fun deleteFile(fileId: Long) = withContext(Dispatchers.IO) {

        val request = Request.Builder()
            .url("$SERVER_BASE_URL/files/$fileId")
            .delete()
            .build()

        val response = client.newCall(request).execute()

        if (!response.isSuccessful) {
            throw RuntimeException("Delete failed: ${response.code}")
        }
    }
}