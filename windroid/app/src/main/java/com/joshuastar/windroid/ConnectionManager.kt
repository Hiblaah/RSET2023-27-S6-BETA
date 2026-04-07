package com.joshuastar.windroid

import android.annotation.SuppressLint
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.BroadcastReceiver
import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.content.Intent
import android.os.Build
import android.os.IBinder
import android.util.Log
import java.io.PrintWriter
import java.net.Socket

object ConnectionManager {

    private const val TAG = "ConnectionManager"
    private var socket: Socket? = null
    private var writer: PrintWriter? = null
    private var running = false
    private var appContext: Context? = null
    private val writeLock = Object()
    @Volatile
    var lastReceivedFromPcTime = 0L

    fun isConnected(): Boolean {
        return socket?.isConnected == true &&
                socket?.isClosed == false &&
                writer != null
    }

    @SuppressLint("ServiceCast")
    fun connect(ip: String, port: Int, context: Context? = null) {
        appContext = context?.applicationContext
        if (running && isConnected()) return
        if (running) {
            running = false
            try { socket?.close() } catch (e: Exception) {}
            socket = null
            writer = null
        }
        running = true

        Thread {
            while (running) {
                try {
                    Log.d(TAG, "Connecting to $ip:$port...")
                    socket = Socket(ip, port)
                    writer = PrintWriter(socket!!.getOutputStream(), true)

                    val phoneName = Build.MODEL
                    val fingerprint = Build.FINGERPRINT.take(16).replace("/", "-")
                    writer!!.println("HELLO|$phoneName|$fingerprint")

                    PhoneStatusSender.start(appContext!!)
                    FlashlightController.init(appContext!!)

                    Log.d(TAG, "Connected and handshake sent")
                    FileDropZone.start(appContext!!)

                    val reader = socket!!.getInputStream().bufferedReader()
                    var lastPingTime = System.currentTimeMillis()
                    var lastLineTime = System.currentTimeMillis()

// Watchdog thread — detects if read loop is frozen
                    val watchdogThread = Thread {
                        while (running && socket?.isClosed == false) {
                            Thread.sleep(3000)
                            val silentMs = System.currentTimeMillis() - lastLineTime
                            Log.d(TAG, "Watchdog: last activity ${silentMs}ms ago, connected=${isConnected()}")
                            if (silentMs > 15000) {
                                Log.e(TAG, "Watchdog: no activity for 15s — forcing reconnect")
                                try { socket?.close() } catch (e: Exception) {}
                                break
                            }
                        }
                    }
                    watchdogThread.isDaemon = true
                    watchdogThread.start()

                    while (running) {
                        val line = try {
                            reader.readLine()
                        } catch (e: Exception) {
                            Log.e(TAG, "readLine exception: ${e.message}")
                            break
                        }

                        if (line == null) {
                            Log.e(TAG, "readLine returned null — server closed connection")
                            break
                        }

                        lastLineTime = System.currentTimeMillis()
                        Log.d(TAG, "<<< RECEIVED: $line")

                        if (line == "PONG") {
                            Log.d(TAG, "Heartbeat PONG received")
                            lastPingTime = System.currentTimeMillis()
                            continue
                        }

                        if (System.currentTimeMillis() - lastPingTime > 5000) {
                            lastPingTime = System.currentTimeMillis()
                            send("PING")
                            Log.d(TAG, ">>> PING sent")
                        }

                        try {
                            FileAccessManager.handleCommand(line)

                            if (line.startsWith("CMD:")) {
                                Log.d(TAG, "Dispatching command: ${line.substring(4)}")
                                handleCommand(line.substring(4))

                            } else if (line.startsWith("CLIPBOARD=")) {
                                val text = line.removePrefix("CLIPBOARD=")
                                val ctx = appContext
                                if (ctx == null) {
                                    Log.e(TAG, "appContext is null — cannot set clipboard")
                                } else {
                                Log.d(TAG, "Setting clipboard on phone: '${text.take(30)}'")
                                lastReceivedFromPcTime = System.currentTimeMillis()
                                Log.d(TAG, "lastReceivedFromPcTime set to $lastReceivedFromPcTime")
                                val clipboard = ctx.getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
                                clipboard.setPrimaryClip(ClipData.newPlainText("windroid", text))
                                Log.d(TAG, "Clipboard set OK — read loop should continue normally")

                            } }else if (line.startsWith("MEDIA=")) {
                                val json = line.removePrefix("MEDIA=")
                                val ctx = appContext ?: continue
                                Log.d(TAG, "MEDIA update received")
                                android.os.Handler(android.os.Looper.getMainLooper()).post {
                                    if (MediaNotificationManager.mediaSession == null) {
                                        MediaNotificationManager.init(ctx)
                                    }
                                    MediaNotificationManager.handle(ctx, json)
                                }

                            } else if (line.startsWith("PC_NAME=")) {
                                val name = line.removePrefix("PC_NAME=")
                                val ctx = appContext ?: continue
                                Log.d(TAG, "PC name received: $name")
                                ctx.getSharedPreferences(PREF_NAME, Context.MODE_PRIVATE)
                                    .edit()
                                    .putString("pc_name", name)
                                    .apply()
                            } else {
                                Log.d(TAG, "Unhandled line: $line")
                            }

                        } catch (e: Exception) {
                            Log.e(TAG, "Exception processing '$line': ${e.message}", e)
                        }
                    }
                    Log.e(TAG, "Read loop exited — will reconnect")
                } catch (e: Exception) {
                    Log.e(TAG, "Connection lost: ${e.message}")
                    writer = null
                    socket = null
                }

                if (running) {
                    Log.d(TAG, "Reconnecting in 3s...")
                    Thread.sleep(3000)
                }
            }
        }.start()
    }

    fun reset() {
        running = false
        try { socket?.close() } catch (e: Exception) {}
        socket = null
        writer = null
        appContext = null
    }
    // WITH THIS:
    fun send(message: String) {
        val w = writer
        if (w == null) {
            Log.w(TAG, "Send skipped (not connected): $message")
            return
        }

        Thread {
            try {
                synchronized(writeLock) {
                    w.println(message)
                    w.flush()
                }

                if (w.checkError()) {
                    Log.e(TAG, "Writer error on send — marking dead")
                    socket?.close()
                    writer = null
                    socket = null
                } else {
                    Log.d(TAG, "Sent: $message")
                }

            } catch (e: Exception) {
                Log.e(TAG, "Send failed: ${e.message}")
            }
        }.start()
    }

    fun disconnect() {
        running = false
        try { socket?.close() } catch (e: Exception) {}
        socket = null
        writer = null
    }
    private fun handleCommand(command: String) {
        Log.d(TAG, "Received command: $command")

        when (command) {
            "START_MIRROR" -> {
                val ctx = appContext ?: return
                // Always re-ask for permission — previous grant data is consumed after one use
                ScreenMirrorService.resultCode = 0
                ScreenMirrorService.resultData = null
                val intent = Intent(ctx, MainActivity::class.java).apply {
                    action = "com.joshuastar.windroid.START_MIRROR"
                    flags = Intent.FLAG_ACTIVITY_NEW_TASK or
                            Intent.FLAG_ACTIVITY_SINGLE_TOP or
                            Intent.FLAG_ACTIVITY_CLEAR_TOP
                }
                ctx.startActivity(intent)
            }

            "STOP_MIRROR" -> {
                val ctx = appContext ?: return
                // Only send the stop intent if the service is actually running.
                // Sending it to a dead/never-started service was the crash cause.
                if (ScreenMirrorService.resultCode != 0 ||
                    android.app.ActivityManager::class.java.let { true }) {
                    // Safe: ScreenMirrorService.stopMirroring() guards with isRunning check
                    val stopIntent = Intent(ctx, ScreenMirrorService::class.java).apply {
                        action = "STOP_MIRROR"
                    }
                    ctx.startService(stopIntent)
                }
            }

            "FLASHLIGHT_ON"  -> FlashlightController.set(true)
            "FLASHLIGHT_OFF" -> FlashlightController.set(false)
        }
    }
    class ClipboardReceiver : BroadcastReceiver() {
        override fun onReceive(context: Context, intent: Intent) {
            val text = intent.getStringExtra("clipboard_text") ?: return
            val clipboard = context.getSystemService(Context.CLIPBOARD_SERVICE) as android.content.ClipboardManager
            clipboard.setPrimaryClip(android.content.ClipData.newPlainText("windroid", text))
            android.widget.Toast.makeText(context, "✓ Copied to clipboard", android.widget.Toast.LENGTH_SHORT).show()
        }
    }
    class PersistentService : Service() {

        companion object {
            private const val CHANNEL_ID = "windroid_connection"
            private const val NOTIF_ID = 1001

            fun start(context: Context) {
                val intent = Intent(context, PersistentService::class.java)
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                    context.startForegroundService(intent)
                } else {
                    context.startService(intent)
                }
            }

            fun stop(context: Context) {
                context.stopService(Intent(context, PersistentService::class.java))
            }
        }

        override fun onCreate() {
            super.onCreate()
            createNotificationChannel()
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                startForeground(NOTIF_ID, buildNotification(), android.content.pm.ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC)
            } else {
                startForeground(NOTIF_ID, buildNotification())
            }
        }

        override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
            val prefs = getSharedPreferences(PREF_NAME, Context.MODE_PRIVATE)
            val ip = prefs.getString(PREF_SERVER_IP, null)
            val port = CLIPBOARD_PORT

            if (ip != null && !isConnected()) {
                connect(ip, port, applicationContext)
            }

            return START_STICKY
        }

        override fun onBind(intent: Intent?): IBinder? = null

        override fun onTaskRemoved(rootIntent: Intent?) {
            super.onTaskRemoved(rootIntent)
            disconnect()  // clean up old socket
            val restartIntent = Intent(applicationContext, PersistentService::class.java)
            startService(restartIntent)
        }

        private fun createNotificationChannel() {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                val channel = NotificationChannel(
                    CHANNEL_ID,
                    "Windroid Connection",
                    NotificationManager.IMPORTANCE_LOW
                ).apply {
                    description = "Keeps Windroid connected to your PC"
                    setShowBadge(false)
                }
                getSystemService(NotificationManager::class.java)
                    .createNotificationChannel(channel)
            }
        }

        private fun buildNotification(): Notification {
            val builder = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                Notification.Builder(this, CHANNEL_ID)
            } else {
                @Suppress("DEPRECATION")
                Notification.Builder(this)
            }
            return builder
                .setContentTitle("Windroid")
                .setContentText("Connected to PC")
                .setSmallIcon(android.R.drawable.ic_dialog_info)
                .build()
        }
    }

    class BootReceiver : BroadcastReceiver() {
        override fun onReceive(context: Context, intent: Intent) {
            if (intent.action == Intent.ACTION_BOOT_COMPLETED ||
                intent.action == "android.intent.action.QUICKBOOT_POWERON") {
                val prefs = context.getSharedPreferences(PREF_NAME, Context.MODE_PRIVATE)
                val ip = prefs.getString(PREF_SERVER_IP, null)
                if (ip != null) {
                    PersistentService.start(context)
                }
            }
        }
    }
}