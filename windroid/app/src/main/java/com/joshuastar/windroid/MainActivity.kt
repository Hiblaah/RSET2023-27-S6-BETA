package com.joshuastar.windroid

import android.Manifest
import android.annotation.SuppressLint
import android.app.*
import android.content.*
import android.content.pm.PackageManager
import android.graphics.*
import android.graphics.drawable.*
import android.os.Build
import android.util.Log
import androidx.activity.ComponentActivity
import android.os.Bundle
import java.io.PrintWriter
import java.net.Socket
import android.widget.*
import android.view.*
import android.view.ViewGroup.LayoutParams.MATCH_PARENT
import android.view.ViewGroup.LayoutParams.WRAP_CONTENT

const val PORT = 1234
private const val REQUEST_MEDIA_PROJECTION = 1003
class MainActivity : ComponentActivity() {
    private var pendingStartMirror = false
    private lateinit var fileSendLauncher: FileSendLauncherHelper
    private lateinit var ipText: TextView
    companion object {
        var serverIp: String = ""
    }
    private var shouldFinishAfterPermissions = false

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        if (intent.action == "com.joshuastar.windroid.START_MIRROR") {
            pendingStartMirror = true
        }
        if (intent.action == "com.joshuastar.windroid.STOP_MIRROR") {
            stopScreenMirror()
        }
    }

    @SuppressLint("UnspecifiedRegisterReceiverFlag")
    override fun onCreate(savedInstanceState: Bundle?) {
        fileSendLauncher = FileSendLauncherHelper(this)
        serverIp = intent.getStringExtra("server_ip")
            ?: getSharedPreferences(PREF_NAME, MODE_PRIVATE).getString(PREF_SERVER_IP, "") ?: ""
        super.onCreate(savedInstanceState)

        createUI()

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            if (checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED) {
                requestPermissions(arrayOf(Manifest.permission.POST_NOTIFICATIONS), 1001)
                return
            }
        }

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            if (!android.provider.Settings.canDrawOverlays(this)) {
                shouldFinishAfterPermissions = true
                requestOverlayPermission()
                return
            }
        }

        checkNotificationListenerPermission()
        startClipboardService()
        ConnectionManager.connect(serverIp, PORT, this)
        val filter = IntentFilter().apply {
            addAction("com.joshuastar.windroid.START_MIRROR")
            addAction("com.joshuastar.windroid.STOP_MIRROR")
        }
        registerReceiver(commandReceiver, filter, RECEIVER_NOT_EXPORTED)
    }

    override fun onResume() {
        super.onResume()
        if (pendingStartMirror) {
            pendingStartMirror = false
            startScreenMirror()
        }
    }
    fun updatePcLabel(pcName: String) {
        ipText.text = "$pcName  •  $serverIp"
    }
    override fun onDestroy() {
        super.onDestroy()
        try { unregisterReceiver(commandReceiver) } catch (_: Exception) {}
        ConnectionManager.disconnect()
    }

    private val commandReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context, intent: Intent) {
            when (intent.action) {
                "com.joshuastar.windroid.START_MIRROR" -> startScreenMirror()
                "com.joshuastar.windroid.STOP_MIRROR"  -> stopScreenMirror()
            }
        }
    }

    private fun dp(value: Int): Int = (value * resources.displayMetrics.density).toInt()

    private fun makeRoundedDrawable(normalColor: Int, pressedColor: Int): StateListDrawable {
        val radius = dp(16).toFloat()

        val normal = GradientDrawable().apply {
            shape = GradientDrawable.RECTANGLE
            cornerRadius = radius
            setColor(normalColor)
        }
        val pressed = GradientDrawable().apply {
            shape = GradientDrawable.RECTANGLE
            cornerRadius = radius
            setColor(pressedColor)
        }

        return StateListDrawable().apply {
            addState(intArrayOf(android.R.attr.state_pressed), pressed)
            addState(intArrayOf(), normal)
        }
    }

    private fun makeGridButton(iconResId: Int?, label: String, onClick: () -> Unit): LinearLayout {
        val card = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            gravity = Gravity.CENTER
            background = makeRoundedDrawable(
                Color.parseColor("#1E1E1E"),
                Color.parseColor("#2A2A2A")
            )
            isClickable = true
            isFocusable = true
            setPadding(dp(12), dp(20), dp(12), dp(16))

            val size = dp(100)
            layoutParams = GridLayout.LayoutParams().apply {
                width = 0
                height = size
                columnSpec = GridLayout.spec(GridLayout.UNDEFINED, 1, 1f)
                setMargins(dp(6), dp(6), dp(6), dp(6))
            }
        }

        if (iconResId != null) {
            val icon = ImageView(this).apply {
                setImageResource(iconResId)
                colorFilter = PorterDuffColorFilter(Color.WHITE, PorterDuff.Mode.SRC_IN)  // ← add this
                layoutParams = LinearLayout.LayoutParams(dp(32), dp(32)).apply {
                    bottomMargin = dp(8)
                    gravity = Gravity.CENTER_HORIZONTAL
                }
                scaleType = ImageView.ScaleType.FIT_CENTER
            }
            card.addView(icon)
        }

        val text = TextView(this).apply {
            text = label
            textSize = 11f
            setTextColor(Color.parseColor("#CCCCCC"))
            gravity = Gravity.CENTER
            layoutParams = LinearLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT)
            maxLines = 2
            ellipsize = android.text.TextUtils.TruncateAt.END
        }
        card.addView(text)

        card.setOnClickListener { onClick() }

        return card
    }

    private fun createUI() {
        val scrollView = ScrollView(this).apply {
            setBackgroundColor(Color.parseColor("#121212"))
        }

        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(16), dp(48), dp(16), dp(24))
        }

        val headerRow = FrameLayout(this).apply {
            layoutParams = LinearLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT).apply {
                bottomMargin = dp(24)
            }
        }

        val headerLayout = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(8), 0, dp(48), 0)
        }

        val title = TextView(this).apply {
            text = "Windroid"
            textSize = 26f
            setTextColor(Color.WHITE)
            typeface = Typeface.create("sans-serif-light", Typeface.NORMAL)
            letterSpacing = 0.05f
        }

        val pcName = getSharedPreferences(PREF_NAME, MODE_PRIVATE).getString("pc_name", "") ?: ""

        ipText = TextView(this).apply {
            text = if (pcName.isNotEmpty()) "$pcName  •  $serverIp" else serverIp.ifEmpty { "Not connected" }
            textSize = 12f
            setTextColor(Color.parseColor("#4CAF50"))
            setPadding(0, dp(4), 0, 0)
        }

        headerLayout.addView(title)
        headerLayout.addView(ipText)

        val forgetIcon = TextView(this).apply {
            text = "✕"
            textSize = 14f
            setTextColor(Color.parseColor("#555555"))
            gravity = Gravity.CENTER
            layoutParams = FrameLayout.LayoutParams(dp(36), dp(36)).apply {
                gravity = Gravity.TOP or Gravity.END
            }
            setOnClickListener {
                AlertDialog.Builder(this@MainActivity)
                    .setTitle("Forget PC")
                    .setMessage("Disconnect and forget \"${pcName.ifEmpty { serverIp }}\"?")
                    .setPositiveButton("Forget") { _, _ ->
                        getSharedPreferences(PREF_NAME, MODE_PRIVATE).edit().clear().apply()
                        ConnectionManager.reset()  // ← add this
                        startActivity(Intent(this@MainActivity, DiscoveryActivity::class.java))
                        finish()
                    }
                    .setNegativeButton("Cancel", null)
                    .show()
            }
        }

        headerRow.addView(headerLayout)
        headerRow.addView(forgetIcon)
        root.addView(headerRow)

        val grid = GridLayout(this).apply {
            columnCount = 3
            layoutParams = LinearLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT)
        }

        // ── Add your buttons here ──
        // Replace R.drawable.ic_mouse etc. with your actual drawable resource names
        // Pass null as iconResId if you haven't added icons yet

        grid.addView(makeGridButton(R.drawable.mouse, "Mouse Control") {
            startActivity(Intent(this, MouseActivity::class.java))
        })

        grid.addView(makeGridButton(R.drawable.lock, "Lock PC") {
            ConnectionManager.send("CMD:LOCK_PC")
            Toast.makeText(this, "Locking PC...", Toast.LENGTH_SHORT).show()
        })

        grid.addView(makeGridButton(R.drawable.signout, "Sign Out") {
            AlertDialog.Builder(this)
                .setTitle("Sign Out")
                .setMessage("This will close all apps and sign out. Continue?")
                .setPositiveButton("Sign Out") { _, _ ->
                    ConnectionManager.send("CMD:SIGNOUT_PC")
                    Toast.makeText(this, "Signing out PC...", Toast.LENGTH_SHORT).show()
                }
                .setNegativeButton("Cancel", null)
                .show()
        })

        grid.addView(makeGridButton(R.drawable.gamecontroller, "Game Controller") {
            startActivity(Intent(this, GameControllerActivity::class.java))
        })

        grid.addView(makeGridButton(R.drawable.folder, "File Access") {
            startActivity(Intent(this, PcFileBrowserActivity::class.java))
        })

        grid.addView(makeGridButton(R.drawable.upload, "Send File") {
            fileSendLauncher.launch()
        })

        grid.addView(makeGridButton(R.drawable.keyboard, "Keyboard") {
            startActivity(Intent(this, KeyboardActivity::class.java))
        })

        root.addView(grid)
        scrollView.addView(root)
        setContentView(scrollView)
    }

    private fun sendHandshake() {
        val phoneName = Build.MODEL
        val fingerprint = Build.FINGERPRINT.take(16).replace("/", "-")
        Thread {
            try {
                val socket = Socket(serverIp, PORT)
                val out = PrintWriter(socket.getOutputStream(), true)
                out.println("HELLO|$phoneName|$fingerprint")
                out.close()
                socket.close()
            } catch (e: Exception) {
                Log.e("HANDSHAKE", "Failed", e)
            }
        }.start()
    }

    private fun checkNotificationListenerPermission() {
        val flat = android.provider.Settings.Secure.getString(contentResolver, "enabled_notification_listeners")
        val enabled = flat?.contains(packageName) == true
        if (!enabled) {
            AlertDialog.Builder(this)
                .setTitle("Enable Notification Access")
                .setMessage("Allow Windroid to read notifications so they can be forwarded to your PC.")
                .setPositiveButton("Open Settings") { _, _ ->
                    startActivity(Intent("android.settings.ACTION_NOTIFICATION_LISTENER_SETTINGS"))
                }
                .setNegativeButton("Skip", null)
                .show()
        }
    }

    private fun startScreenMirror() {
        if (ScreenMirrorService.resultCode == 0 || ScreenMirrorService.resultData == null) {
            val mediaProjectionManager = getSystemService(MEDIA_PROJECTION_SERVICE)
                    as android.media.projection.MediaProjectionManager
            startActivityForResult(mediaProjectionManager.createScreenCaptureIntent(), REQUEST_MEDIA_PROJECTION)
            return
        }
        val intent = Intent(this, ScreenMirrorService::class.java)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) startForegroundService(intent)
        else startService(intent)
    }

    private fun stopScreenMirror() {
        val intent = Intent(this, ScreenMirrorService::class.java).apply { action = "STOP_MIRROR" }
        startService(intent)
        Toast.makeText(this, "Screen mirror stopped", Toast.LENGTH_SHORT).show()
    }

    private fun requestOverlayPermission() {
        AlertDialog.Builder(this)
            .setTitle("Display Over Other Apps Permission")
            .setMessage("This app needs permission to display over other apps to read clipboard in the background.")
            .setPositiveButton("Grant Permission") { _, _ ->
                val intent = Intent(
                    android.provider.Settings.ACTION_MANAGE_OVERLAY_PERMISSION,
                    android.net.Uri.parse("package:$packageName")
                )
                startActivityForResult(intent, 1002)
            }
            .setNegativeButton("Cancel") { _, _ ->
                Toast.makeText(this, "Overlay permission is required for clipboard sync", Toast.LENGTH_LONG).show()
            }
            .setCancelable(false)
            .show()
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)

        if (requestCode == REQUEST_MEDIA_PROJECTION) {
            if (resultCode == RESULT_OK && data != null) {
                // Store fresh permission — service will consume & clear it on use
                ScreenMirrorService.resultCode = resultCode
                ScreenMirrorService.resultData = data

                val intent = Intent(this, ScreenMirrorService::class.java)
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) startForegroundService(intent)
                else startService(intent)

                Toast.makeText(this, "✓ Screen mirroring started", Toast.LENGTH_SHORT).show()
            } else {
                // User denied — clear any stale permission and tell PC to reset its toggle
                ScreenMirrorService.resultCode = 0
                ScreenMirrorService.resultData = null
                ConnectionManager.send("MIRROR_STOPPED")
                Toast.makeText(this, "✗ Screen mirroring permission denied", Toast.LENGTH_SHORT).show()
            }
        }

        if (requestCode == 1002) {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
                if (android.provider.Settings.canDrawOverlays(this)) {
                    startClipboardService()
                    if (shouldFinishAfterPermissions) {
                        Toast.makeText(this, "✓ All permissions granted", Toast.LENGTH_SHORT).show()
                    }
                } else {
                    Toast.makeText(this, "Permission denied. Clipboard sync may not work.", Toast.LENGTH_LONG).show()
                }
            }
        }
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == 1001 && grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
                if (!android.provider.Settings.canDrawOverlays(this)) {
                    requestOverlayPermission()
                    return
                }
            }
            startClipboardService()
        }
    }

    private fun startClipboardService() {
        val intent = Intent(this, ClipboardService::class.java)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) startForegroundService(intent)
        else startService(intent)
    }
}