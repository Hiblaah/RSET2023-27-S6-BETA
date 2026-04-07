package com.joshuastar.windroid

import android.app.*
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.PixelFormat
import android.hardware.display.DisplayManager
import android.hardware.display.VirtualDisplay
import android.media.Image
import android.media.ImageReader
import android.media.projection.MediaProjection
import android.media.projection.MediaProjectionManager
import android.os.Build
import android.os.Handler
import android.os.HandlerThread
import android.os.IBinder
import android.os.Looper
import android.util.Log
import androidx.core.app.NotificationCompat
import java.io.ByteArrayOutputStream
import java.io.DataOutputStream
import java.net.Socket
import java.nio.ByteBuffer
import java.util.concurrent.atomic.AtomicBoolean

const val MIRROR_PORT = 3333
private const val MIRROR_CHANNEL_ID = "screen_mirror"
private const val ACTION_STOP_MIRROR = "STOP_MIRROR"

class ScreenMirrorService : Service() {

    private var mediaProjection: MediaProjection? = null
    private var virtualDisplay: VirtualDisplay? = null
    private var imageReader: ImageReader? = null
    private var isRunning = false
    private val handler = Handler(Looper.getMainLooper())
    private var imageHandlerThread: HandlerThread? = null
    private var imageHandler: Handler? = null
    private val isSending = AtomicBoolean(false)

    companion object {
        // These MUST be cleared after use — MediaProjection intent data is single-use only.
        // MainActivity sets them fresh each time the user accepts the permission popup.
        var resultCode: Int = 0
        var resultData: Intent? = null
    }

    override fun onCreate() {
        super.onCreate()
        createNotificationChannel()
        startForeground(2, createNotification())

        imageHandlerThread = HandlerThread("ImageProcessing").apply { start() }
        imageHandler = Handler(imageHandlerThread!!.looper)
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent?.action == ACTION_STOP_MIRROR) {
            // PC-initiated or notification button stop — no need to notify PC back
            stopMirroring(notifyPc = false)
            return START_NOT_STICKY
        }

        if (!isRunning) {
            startMirroring()
        }

        return START_NOT_STICKY
    }

    private fun startMirroring() {
        val code = resultCode
        val data = resultData

        if (code == 0 || data == null) {
            Log.e("MIRROR", "No media projection permission — user likely denied")
            notifyPcStopped()
            stopSelf()
            return
        }

        // Consume immediately — this data is single-use, clear before any chance of reuse
        resultCode = 0
        resultData = null

        try {
            val mediaProjectionManager =
                getSystemService(MEDIA_PROJECTION_SERVICE) as MediaProjectionManager

            mediaProjection = mediaProjectionManager.getMediaProjection(code, data)

            if (mediaProjection == null) {
                Log.e("MIRROR", "getMediaProjection returned null")
                notifyPcStopped()
                stopSelf()
                return
            }

            mediaProjection!!.registerCallback(object : MediaProjection.Callback() {
                override fun onStop() {
                    Log.d("MIRROR", "MediaProjection stopped externally")
                    stopMirroring(notifyPc = true)
                }
            }, handler)

            val metrics = resources.displayMetrics
            val scaledWidth  = metrics.widthPixels  / 3
            val scaledHeight = metrics.heightPixels / 3
            val density      = metrics.densityDpi

            imageReader = ImageReader.newInstance(
                scaledWidth, scaledHeight,
                PixelFormat.RGBA_8888,
                2
            )

            virtualDisplay = mediaProjection!!.createVirtualDisplay(
                "ScreenMirror",
                scaledWidth, scaledHeight, density,
                DisplayManager.VIRTUAL_DISPLAY_FLAG_AUTO_MIRROR,
                imageReader?.surface, null, null
            )

            isRunning = true

            imageReader?.setOnImageAvailableListener({ reader ->
                if (isSending.get()) {
                    reader.acquireLatestImage()?.close()
                    return@setOnImageAvailableListener
                }

                var image: Image? = null
                try {
                    image = reader.acquireLatestImage()
                    if (image != null) {
                        isSending.set(true)
                        val bitmap = convertToBitmap(image)
                        image.close()
                        image = null
                        sendBitmapToServer(bitmap)
                    }
                } catch (e: Exception) {
                    Log.e("MIRROR", "Error processing image: ${e.message}")
                    image?.close()
                    isSending.set(false)
                }
            }, imageHandler)

            Log.d("MIRROR", "Screen mirroring started (${scaledWidth}x${scaledHeight})")

        } catch (e: Exception) {
            Log.e("MIRROR", "Error starting mirroring", e)
            notifyPcStopped()
            stopSelf()
        }
    }

    private fun notifyPcStopped() {
        Thread {
            try {
                ConnectionManager.send("MIRROR_STOPPED")
                Log.d("MIRROR", "Notified PC: MIRROR_STOPPED")
            } catch (e: Exception) {
                Log.w("MIRROR", "Could not notify PC of stop: ${e.message}")
            }
        }.start()
    }

    private fun sendBitmapToServer(bitmap: Bitmap) {
        Thread {
            var socket: Socket? = null
            try {
                val outputStream = ByteArrayOutputStream()
                bitmap.compress(Bitmap.CompressFormat.JPEG, 50, outputStream)
                val jpegData = outputStream.toByteArray()

                socket = Socket(MainActivity.serverIp, MIRROR_PORT)
                socket.soTimeout = 2000
                socket.tcpNoDelay = true

                val dos = DataOutputStream(socket.outputStream)
                dos.writeBytes("FRAME:")
                dos.writeInt(jpegData.size)
                dos.write(jpegData)
                dos.flush()

            } catch (e: Exception) {
                Log.w("MIRROR", "Send failed (will retry next frame): ${e.message}")
            } finally {
                bitmap.recycle()
                isSending.set(false)
                try { socket?.close() } catch (_: Exception) {}
            }
        }.start()
    }

    private fun convertToBitmap(image: Image): Bitmap {
        val planes      = image.planes
        val buffer      = planes[0].buffer
        val pixelStride = planes[0].pixelStride
        val rowStride   = planes[0].rowStride
        val rowPadding  = rowStride - pixelStride * image.width

        val byteBuffer = ByteBuffer.allocate(buffer.remaining())
        byteBuffer.put(buffer)
        byteBuffer.rewind()

        val bitmap = Bitmap.createBitmap(
            image.width + rowPadding / pixelStride,
            image.height,
            Bitmap.Config.ARGB_8888
        )
        bitmap.copyPixelsFromBuffer(byteBuffer)

        return if (rowPadding == 0) {
            bitmap
        } else {
            val cropped = Bitmap.createBitmap(bitmap, 0, 0, image.width, image.height)
            bitmap.recycle()
            cropped
        }
    }

    private fun stopMirroring(notifyPc: Boolean = false) {
        // Guard: if we were never fully started, just clean up and exit safely
        val wasRunning = isRunning
        isRunning = false
        isSending.set(false)

        if (notifyPc && wasRunning) notifyPcStopped()

        try { imageReader?.close();       imageReader     = null } catch (_: Exception) {}
        try { virtualDisplay?.release();  virtualDisplay  = null } catch (_: Exception) {}
        try { mediaProjection?.stop();    mediaProjection = null } catch (_: Exception) {}
        try {
            imageHandlerThread?.quitSafely()
            imageHandlerThread = null
            imageHandler = null
        } catch (_: Exception) {}

        stopForeground(STOP_FOREGROUND_REMOVE)
        stopSelf()

        Log.d("MIRROR", "Screen mirroring stopped (wasRunning=$wasRunning, notifyPc=$notifyPc)")
    }

    override fun onDestroy() {
        if (isRunning) stopMirroring(notifyPc = true)
        super.onDestroy()
    }

    override fun onBind(intent: Intent?): IBinder? = null

    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                MIRROR_CHANNEL_ID,
                "Screen Mirroring",
                NotificationManager.IMPORTANCE_LOW
            )
            getSystemService(NotificationManager::class.java).createNotificationChannel(channel)
        }
    }

    private fun createNotification(): Notification {
        val stopIntent = Intent(this, ScreenMirrorService::class.java).apply {
            action = ACTION_STOP_MIRROR
        }
        val stopPendingIntent = PendingIntent.getService(
            this, 0, stopIntent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )
        return NotificationCompat.Builder(this, MIRROR_CHANNEL_ID)
            .setContentTitle("Screen Mirroring Active")
            .setContentText("Your screen is being mirrored to PC")
            .setSmallIcon(R.mipmap.ic_launcher)
            .setOngoing(true)
            .addAction(android.R.drawable.ic_media_pause, "Stop", stopPendingIntent)
            .build()
    }
}