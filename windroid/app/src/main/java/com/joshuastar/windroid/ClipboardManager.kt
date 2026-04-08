package com.joshuastar.windroid

import android.accessibilityservice.AccessibilityService
import android.accessibilityservice.GestureDescription
import android.annotation.SuppressLint
import android.app.*
import android.content.*
import android.graphics.Path
import android.graphics.PixelFormat
import android.os.*
import android.service.quicksettings.Tile
import android.service.quicksettings.TileService
import android.util.Log
import android.view.View
import android.view.WindowManager
import android.view.accessibility.AccessibilityEvent
import android.view.accessibility.AccessibilityNodeInfo
import android.widget.FrameLayout
import androidx.core.app.NotificationCompat
import java.io.File

private const val CHANNEL_ID = "clipboard_sync"
const val ACTION_SYNC_CLIPBOARD = "SYNC_CLIPBOARD"

// ─────────────────────────────────────────────
// Clipboard Sender (used only by Accessibility)
// ─────────────────────────────────────────────
object ClipboardSender {
    fun readAndSend(context: Context) {

        val clipboard =
            context.getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager

        val clip = clipboard.primaryClip
        if (clip == null) {
            Log.e("CLIPBOARD", "Clipboard is NULL (background restriction)")
            return
        }

        if (clip.itemCount == 0) return

        val text = clip.getItemAt(0).coerceToText(context).toString()
        if (text.isBlank()) return

        Log.d("CLIPBOARD", "Sending: $text")

// With this:
        val timeSinceReceived = System.currentTimeMillis() - ConnectionManager.lastReceivedFromPcTime
        if (timeSinceReceived < 2000) {
            Log.d("CLIPBOARD", "Ignored clipboard (came from PC, ${timeSinceReceived}ms ago)")
            return
        }

        Log.d("CLIPBOARD", "About to send. isConnected=${ConnectionManager.isConnected()}, timeSincePC=${System.currentTimeMillis() - ConnectionManager.lastReceivedFromPcTime}ms")
        ConnectionManager.send("CLIPBOARD=$text")
        Log.d("CLIPBOARD", "Send call returned")
    }
}

// ─────────────────────────────────────────────
// Foreground Service (keeps connection alive)
// ─────────────────────────────────────────────
class ClipboardService : Service() {

    override fun onCreate() {
        super.onCreate()
        startForeground(1, createNotification())

        val ip = getSharedPreferences(PREF_NAME, MODE_PRIVATE)
            .getString(PREF_SERVER_IP, "") ?: ""
        if (ip.isNotEmpty()) {
            ConnectionManager.connect(ip, PORT, this)
        }
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        val ip = intent?.getStringExtra("server_ip")
        if (ip != null && ip.isNotEmpty()) {
            ConnectionManager.connect(ip, PORT, this)
        }
        return START_STICKY  // ← this makes Android restart the service if killed
    }

    override fun onBind(intent: Intent?): IBinder? = null

    private fun createNotification(): Notification {

        val manager = getSystemService(NOTIFICATION_SERVICE) as NotificationManager

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                CHANNEL_ID,
                "Clipboard Sync",
                NotificationManager.IMPORTANCE_LOW
            )
            manager.createNotificationChannel(channel)
        }

        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle("Windroid Running")
            .setContentText("Connected")
            .setSmallIcon(R.mipmap.ic_launcher)
            .setOngoing(true)
            .build()
    }
}

// ─────────────────────────────────────────────
// Quick Settings Tile
// ─────────────────────────────────────────────
class ClipboardTileService : TileService() {

    override fun onStartListening() {
        super.onStartListening()
        qsTile?.state = Tile.STATE_ACTIVE
        qsTile?.label = "Send Clipboard"
        qsTile?.updateTile()
    }

    @SuppressLint("StartActivityAndCollapseDeprecated")
    override fun onClick() {
        super.onClick()
        val intent = Intent(this, ClipboardSyncActivity::class.java).apply {
            flags = Intent.FLAG_ACTIVITY_NEW_TASK
        }
        val pendingIntent = PendingIntent.getActivity(
            this, 0, intent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.UPSIDE_DOWN_CAKE) {
            startActivityAndCollapse(pendingIntent)
        } else {
            @Suppress("DEPRECATION")
            startActivityAndCollapse(intent)
        }
    }


}

// ─────────────────────────────────────────────
// Accessibility Service (REAL clipboard reader)
// ─────────────────────────────────────────────
class ClipboardAccessibilityService : AccessibilityService() {

    private var isRunning = true
    private var lastTriggerTime = 0L
    private lateinit var windowManager: WindowManager
    private var overlayView: View? = null
    private val mainHandler = Handler(Looper.getMainLooper())

    override fun onServiceConnected() {
        super.onServiceConnected()

        windowManager = getSystemService(WINDOW_SERVICE) as WindowManager
        Log.d("ACCESSIBILITY", "Service connected")

        Thread {
            val triggerFile = File(filesDir, "clipboard_trigger")

            while (isRunning) {

                if (triggerFile.exists()) {
                    val now = System.currentTimeMillis()

                    if (now - lastTriggerTime > 1000) {
                        lastTriggerTime = now
                        triggerFile.delete()

                        mainHandler.post {
                            showOverlayAndReadClipboard()
                        }
                    } else {
                        triggerFile.delete()
                    }
                }

                Thread.sleep(150)
            }
        }.start()
    }

    private fun showOverlayAndReadClipboard() {
        try {
            val params = WindowManager.LayoutParams(
                WindowManager.LayoutParams.MATCH_PARENT,
                WindowManager.LayoutParams.MATCH_PARENT,
                WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY,
                // Remove NOT_FOCUSABLE so the overlay can gain focus
                WindowManager.LayoutParams.FLAG_LAYOUT_NO_LIMITS or
                        WindowManager.LayoutParams.FLAG_LAYOUT_IN_SCREEN,
                PixelFormat.TRANSLUCENT
            )

            overlayView = FrameLayout(this).apply {
                alpha = 0.01f
                setBackgroundColor(0x01000000)
                isFocusable = true
                isFocusableInTouchMode = true
            }

            windowManager.addView(overlayView, params)
            overlayView?.requestFocus()

            // Increased delay to ensure focus is gained
            mainHandler.postDelayed({
                ClipboardSender.readAndSend(this)

                mainHandler.postDelayed({
                    overlayView?.let {
                        try { windowManager.removeView(it) } catch (_: Exception) {}
                    }
                    overlayView = null
                }, 150)

            }, 800) // increased from 300 to 800

        } catch (e: Exception) {
            Log.e("ACCESSIBILITY", "Overlay failed", e)
        }
    }

    override fun onAccessibilityEvent(event: AccessibilityEvent?) {}
    override fun onInterrupt() {}

    override fun onDestroy() {
        isRunning = false
        overlayView?.let {
            try { windowManager.removeView(it) } catch (_: Exception) {}
        }
        super.onDestroy()
    }
}
class ClipboardSyncActivity : Activity() {

    private var hasSent = false

    // WITH THIS — just remove the Thread block entirely:
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setTaskDescription(ActivityManager.TaskDescription("", null, 0x00000000))
        window.addFlags(WindowManager.LayoutParams.FLAG_NOT_TOUCHABLE)
        window.setDimAmount(0f)
        // No connect here — PersistentService already maintains the connection

        Thread {
            val ip = getSharedPreferences(PREF_NAME, MODE_PRIVATE)
                .getString(PREF_SERVER_IP, "") ?: ""
            if (ip.isNotEmpty()) {
                ConnectionManager.connect(ip, PORT)
            }
        }.start()
    }

    override fun onWindowFocusChanged(hasFocus: Boolean) {
        super.onWindowFocusChanged(hasFocus)
        if (hasFocus && !hasSent) {
            hasSent = true
            ClipboardSender.readAndSend(this)
            finish()
        }
    }
}