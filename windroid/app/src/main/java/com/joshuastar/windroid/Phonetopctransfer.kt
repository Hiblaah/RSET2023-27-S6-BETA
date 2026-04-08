package com.joshuastar.windroid

import android.app.NotificationChannel
import android.app.NotificationManager
import android.content.Context
import android.content.Intent
import android.net.Uri
import android.os.Build
import android.provider.OpenableColumns
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.result.ActivityResultLauncher
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.app.NotificationCompat
import java.io.DataOutputStream

object PhoneToPcTransfer {

    private const val TAG = "PhoneToPcTransfer"
    private const val FILE_TRANSFER_PORT = 6790
    private const val CHANNEL_ID = "windroid_phone_to_pc"
    private const val PROGRESS_NOTIF_ID = 9002

    fun sendFileToPc(context: Context, uri: Uri) {
        val prefs = context.getSharedPreferences(PREF_NAME, Context.MODE_PRIVATE)
        val ip = prefs.getString(PREF_SERVER_IP, null)
        if (ip == null) {
            Log.e(TAG, "No PC IP saved")
            return
        }

        createChannel(context)

        val fileName = resolveFileName(context, uri) ?: "file_${System.currentTimeMillis()}"
        val fileSize = resolveFileSize(context, uri)

        Thread {
            val notifManager = context.getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
            try {
                val inputStream = context.contentResolver.openInputStream(uri)
                    ?: throw Exception("Cannot open file")

                val socket = java.net.Socket(ip, FILE_TRANSFER_PORT)
                val dos = DataOutputStream(socket.getOutputStream())

                val nameBytes = fileName.toByteArray(Charsets.UTF_8)
                dos.writeInt(nameBytes.size)
                dos.write(nameBytes)
                dos.writeLong(fileSize)
                dos.flush()

                val buffer = ByteArray(65536)
                var sent = 0L
                var lastNotifAt = 0L
                var read: Int

                while (inputStream.read(buffer).also { read = it } != -1) {
                    dos.write(buffer, 0, read)
                    sent += read

                    if (sent - lastNotifAt > 512 * 1024 || sent == fileSize) {
                        lastNotifAt = sent
                        val pct = if (fileSize > 0) ((sent * 100) / fileSize).toInt() else 0
                        showProgressNotif(context, notifManager, fileName, pct, sent, fileSize)
                    }
                }

                dos.flush()
                inputStream.close()
                socket.close()

                showDoneNotif(context, notifManager, fileName)
                Log.d(TAG, "Sent $fileName to PC")

            } catch (e: Exception) {
                Log.e(TAG, "Send failed: ${e.message}")
                notifManager.cancel(PROGRESS_NOTIF_ID)
                showErrorNotif(context, notifManager, fileName)
            }
        }.start()
    }

    private fun resolveFileName(context: Context, uri: Uri): String? {
        var name: String? = null
        context.contentResolver.query(uri, null, null, null, null)?.use { cursor ->
            val col = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME)
            if (cursor.moveToFirst() && col >= 0) name = cursor.getString(col)
        }
        return name ?: uri.lastPathSegment
    }

    private fun resolveFileSize(context: Context, uri: Uri): Long {
        var size = 0L
        context.contentResolver.query(uri, null, null, null, null)?.use { cursor ->
            val col = cursor.getColumnIndex(OpenableColumns.SIZE)
            if (cursor.moveToFirst() && col >= 0) size = cursor.getLong(col)
        }
        return size
    }

    private fun showProgressNotif(
        context: Context,
        manager: NotificationManager,
        fileName: String,
        pct: Int,
        sent: Long,
        total: Long
    ) {
        val notif = NotificationCompat.Builder(context, CHANNEL_ID)
            .setSmallIcon(android.R.drawable.stat_sys_upload)
            .setContentTitle("Sending to PC")
            .setContentText("$fileName  •  ${toReadable(sent)} / ${toReadable(total)}")
            .setProgress(100, pct, total == 0L)
            .setOngoing(true)
            .setOnlyAlertOnce(true)
            .setSilent(true)
            .build()

        manager.notify(PROGRESS_NOTIF_ID, notif)
    }

    private fun showDoneNotif(context: Context, manager: NotificationManager, fileName: String) {
        manager.cancel(PROGRESS_NOTIF_ID)
        val notif = NotificationCompat.Builder(context, CHANNEL_ID)
            .setSmallIcon(android.R.drawable.stat_sys_upload_done)
            .setContentTitle("File sent to PC")
            .setContentText(fileName)
            .setAutoCancel(true)
            .build()
        manager.notify(fileName.hashCode(), notif)
    }

    private fun showErrorNotif(context: Context, manager: NotificationManager, fileName: String) {
        val notif = NotificationCompat.Builder(context, CHANNEL_ID)
            .setSmallIcon(android.R.drawable.stat_notify_error)
            .setContentTitle("Failed to send file")
            .setContentText(fileName)
            .setAutoCancel(true)
            .build()
        manager.notify(fileName.hashCode() + 1, notif)
    }

    private fun createChannel(context: Context) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                CHANNEL_ID,
                "Phone to PC Transfer",
                NotificationManager.IMPORTANCE_LOW
            ).apply {
                setShowBadge(false)
            }
            (context.getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager)
                .createNotificationChannel(channel)
        }
    }

    private fun toReadable(bytes: Long): String {
        if (bytes < 1024) return "$bytes B"
        if (bytes < 1024 * 1024) return "%.1f KB".format(bytes / 1024.0)
        return "%.1f MB".format(bytes / (1024.0 * 1024))
    }
}

class FileSendLauncherHelper(activity: ComponentActivity) {

    private val launcher: ActivityResultLauncher<Array<String>> =
        activity.registerForActivityResult(ActivityResultContracts.OpenMultipleDocuments()) { uris ->
            if (uris.isNullOrEmpty()) return@registerForActivityResult
            val ctx = activity.applicationContext
            for (uri in uris) {
                PhoneToPcTransfer.sendFileToPc(ctx, uri)
            }
        }

    fun launch() {
        launcher.launch(arrayOf("*/*"))
    }
}