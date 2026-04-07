package com.joshuastar.windroid

import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.Context
import android.content.Intent
import android.os.Build
import android.os.Environment
import android.util.Log
import androidx.core.app.NotificationCompat
import androidx.core.content.FileProvider
import android.media.MediaScannerConnection
import java.io.DataInputStream
import java.io.File
import java.io.FileOutputStream
import java.net.ServerSocket

object FileDropZone {

    private const val TAG = "FileDropZone"
    private const val FILE_TRANSFER_PORT = 6789
    private const val CHANNEL_ID = "windroid_file_transfer"
    private const val PROGRESS_NOTIF_ID = 9001

    private var running = false

    fun start(context: Context) {
        if (running) return
        running = true
        createChannel(context)

        Thread {
            try {
                val serverSocket = ServerSocket(FILE_TRANSFER_PORT)
                Log.d(TAG, "Listening on port $FILE_TRANSFER_PORT")

                while (running) {
                    val client = serverSocket.accept()
                    Thread { receiveFile(client, context.applicationContext) }.start()
                }

                serverSocket.close()
            } catch (e: Exception) {
                Log.e(TAG, "Server error: ${e.message}")
            }
        }.start()
    }

    fun stop() {
        running = false
    }

    private fun receiveFile(socket: java.net.Socket, context: Context) {
        val notifManager = context.getSystemService(Context.NOTIFICATION_SERVICE)
                as NotificationManager

        try {
            val dis = DataInputStream(socket.getInputStream())

            val nameLen = dis.readInt()
            val nameBytes = ByteArray(nameLen)
            dis.readFully(nameBytes)
            val fileName = String(nameBytes, Charsets.UTF_8)
            val fileSize = dis.readLong()

            Log.d(TAG, "Receiving: $fileName ($fileSize bytes)")

            val downloadsDir = Environment.getExternalStoragePublicDirectory(
                Environment.DIRECTORY_DOWNLOADS
            )
            downloadsDir.mkdirs()

            val outFile = File(downloadsDir, fileName)
            val fos = FileOutputStream(outFile)
            val buffer = ByteArray(65536)
            var received = 0L
            var lastNotifAt = 0L

            while (received < fileSize) {
                val toRead = minOf(buffer.size.toLong(), fileSize - received).toInt()
                val read = dis.read(buffer, 0, toRead)
                if (read == -1) break
                fos.write(buffer, 0, read)
                received += read

                if (received - lastNotifAt > 512 * 1024 || received == fileSize) {
                    lastNotifAt = received
                    val pct = ((received * 100) / fileSize).toInt()
                    showProgressNotif(context, notifManager, fileName, pct, received, fileSize)
                }
            }

            fos.flush()
            fos.close()
            socket.close()

            MediaScannerConnection.scanFile(context, arrayOf(outFile.absolutePath), null, null)
            showDoneNotif(context, notifManager, fileName, outFile)
            Log.d(TAG, "Saved to ${outFile.absolutePath}")

        } catch (e: Exception) {
            Log.e(TAG, "Receive error: ${e.message}")
            notifManager.cancel(PROGRESS_NOTIF_ID)
        }
    }

    private fun showProgressNotif(
        context: Context,
        manager: NotificationManager,
        fileName: String,
        pct: Int,
        received: Long,
        total: Long
    ) {
        val notif = NotificationCompat.Builder(context, CHANNEL_ID)
            .setSmallIcon(android.R.drawable.stat_sys_download)
            .setContentTitle("Receiving file")
            .setContentText("$fileName  •  ${toReadable(received)} / ${toReadable(total)}")
            .setProgress(100, pct, false)
            .setOngoing(true)
            .setOnlyAlertOnce(true)
            .setSilent(true)
            .build()

        manager.notify(PROGRESS_NOTIF_ID, notif)
    }

    private fun showDoneNotif(
        context: Context,
        manager: NotificationManager,
        fileName: String,
        file: File
    ) {
        manager.cancel(PROGRESS_NOTIF_ID)

        val mime = android.webkit.MimeTypeMap.getSingleton()
            .getMimeTypeFromExtension(file.extension) ?: "*/*"

        val uri = FileProvider.getUriForFile(
            context, "${context.packageName}.provider", file
        )

        val openIntent = Intent(Intent.ACTION_VIEW).apply {
            setDataAndType(uri, mime)
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        }

        val pendingIntent = PendingIntent.getActivity(
            context, file.hashCode(), openIntent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )

        val notif = NotificationCompat.Builder(context, CHANNEL_ID)
            .setSmallIcon(android.R.drawable.stat_sys_download_done)
            .setContentTitle("File received")
            .setContentText(fileName)
            .setContentIntent(pendingIntent)
            .setAutoCancel(true)
            .build()

        manager.notify(file.hashCode(), notif)
    }

    private fun createChannel(context: Context) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                CHANNEL_ID,
                "File Transfers",
                NotificationManager.IMPORTANCE_LOW
            ).apply {
                description = "Shows file transfer progress from PC"
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