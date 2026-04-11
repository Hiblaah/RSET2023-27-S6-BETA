package com.joshuastar.windroid

import android.content.Intent
import android.graphics.Color
import android.graphics.Typeface
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.provider.Settings
import android.util.Base64
import android.view.Gravity
import android.view.ViewGroup.LayoutParams.MATCH_PARENT
import android.view.ViewGroup.LayoutParams.WRAP_CONTENT
import android.widget.*
import androidx.activity.ComponentActivity
import java.io.File

// ─────────────────────────────────────────────
// Holds a reference to the active browser activity
// so FileAccessManager can call updateList()
// ─────────────────────────────────────────────
object BrowserHolder {
    var activity: PcFileBrowserActivity? = null
}

// ─────────────────────────────────────────────
// Handles all file-related socket messages
// (both PC ↔ Android directions)
// ─────────────────────────────────────────────
object FileAccessManager {

    fun handleCommand(message: String) {

        // PC requesting Android folder listing
        if (message.startsWith("ANDROID_REQ_LIST|")) {
            val path = message.removePrefix("ANDROID_REQ_LIST|")
            val dir = File(path)
            if (!dir.exists() || !dir.isDirectory) return

            val builder = StringBuilder("ANDROID_RES_LIST|")
            dir.listFiles()?.forEach {
                builder.append(it.name)
                    .append(",")
                    .append(if (it.isDirectory) "DIR" else "FILE")
                    .append(";")
            }
            ConnectionManager.send(builder.toString())
        }

        // PC requesting a file download from Android
        else if (message.startsWith("ANDROID_REQ_DOWNLOAD|")) {
            val path = message.removePrefix("ANDROID_REQ_DOWNLOAD|")
            val file = File(path)
            if (!file.exists() || !file.isFile) return

            val base64 = Base64.encodeToString(file.readBytes(), Base64.NO_WRAP)
            ConnectionManager.send("ANDROID_RES_FILE|${file.name}|$base64")
        }

        // Android receiving a file sent from the PC
        else if (message.startsWith("FILE_RES_FILE|")) {
            val parts = message.split("|", limit = 3)
            val name = parts[1]
            val bytes = Base64.decode(parts[2], Base64.DEFAULT)
            val dest = File(
                Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS),
                name
            )
            dest.writeBytes(bytes)
        }

        // PC sending folder listing back to Android browser
        else if (message.startsWith("FILE_RES_LIST|")) {
            val raw = message.removePrefix("FILE_RES_LIST|")
            val items = raw.split(";")
                .filter { it.isNotBlank() }
                .map {
                    val parts = it.split(",", limit = 2)
                    Pair(parts[0], if (parts.size > 1) parts[1] else "FILE")
                }
            BrowserHolder.activity?.updateList(items)
        }
    }
}

// ─────────────────────────────────────────────
// Activity: browse PC files from Android
// ─────────────────────────────────────────────
class PcFileBrowserActivity : ComponentActivity() {

    private lateinit var listLayout: LinearLayout
    private lateinit var statusDot: TextView
    private lateinit var statusText: TextView

    private var currentPath = ""
    private val pathHistory = mutableListOf<String>()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        createUI()
        BrowserHolder.activity = this

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            if (!Environment.isExternalStorageManager()) {
                val intent = Intent(Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION).apply {
                    data = android.net.Uri.parse("package:$packageName")
                }
                startActivity(intent)
                return
            }
        }

        requestFolder("")
    }

    override fun onDestroy() {
        super.onDestroy()
        BrowserHolder.activity = null
    }

    private fun dp(v: Int): Int = (v * resources.displayMetrics.density).toInt()

    private fun createUI() {

        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(Color.parseColor("#121212"))
            layoutParams = LinearLayout.LayoutParams(MATCH_PARENT, MATCH_PARENT)
        }

        val topBar = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setPadding(dp(16), dp(48), dp(16), dp(16))
        }

        val backBtn = TextView(this).apply {
            text = "←"
            textSize = 22f
            setTextColor(Color.WHITE)
            setPadding(0, 0, dp(16), 0)
            setOnClickListener {
                if (pathHistory.isNotEmpty()) {
                    requestFolder(pathHistory.removeAt(pathHistory.lastIndex))
                } else {
                    finish()
                }
            }
        }

        val title = TextView(this).apply {
            text = "PC File Browser"
            textSize = 20f
            setTextColor(Color.WHITE)
            typeface = Typeface.create("sans-serif-light", Typeface.NORMAL)
        }

        topBar.addView(backBtn)
        topBar.addView(title)
        root.addView(topBar)

        val statusRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setPadding(dp(20), dp(8), dp(20), dp(24))
        }

        statusDot = TextView(this).apply {
            text = "●"
            textSize = 10f
            setTextColor(
                if (ConnectionManager.isConnected())
                    Color.parseColor("#4CAF50")
                else
                    Color.parseColor("#FF5555")
            )
            setPadding(0, 0, dp(8), 0)
        }

        statusText = TextView(this).apply {
            text = if (ConnectionManager.isConnected()) "Connected to PC" else "Not connected"
            textSize = 12f
            setTextColor(Color.parseColor("#888888"))
        }

        statusRow.addView(statusDot)
        statusRow.addView(statusText)
        root.addView(statusRow)

        val divider = LinearLayout(this).apply {
            setBackgroundColor(Color.parseColor("#222222"))
            layoutParams = LinearLayout.LayoutParams(MATCH_PARENT, dp(1)).apply {
                bottomMargin = dp(16)
            }
        }
        root.addView(divider)

        val scroll = ScrollView(this)
        listLayout = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(16), dp(8), dp(16), dp(16))
        }

        scroll.addView(listLayout)
        root.addView(scroll)

        setContentView(root)
    }

    private fun requestFolder(path: String) {
        currentPath = path
        if (path.isEmpty()) {
            ConnectionManager.send("FILE_REQ_DRIVES")
        } else {
            ConnectionManager.send("FILE_REQ_LIST|$path")
        }
    }

    fun updateList(items: List<Pair<String, String>>) {

        runOnUiThread {

            listLayout.removeAllViews()

            items.forEach { (name, type) ->

                val item = LinearLayout(this).apply {
                    orientation = LinearLayout.HORIZONTAL
                    gravity = Gravity.CENTER_VERTICAL
                    setPadding(dp(16), dp(14), dp(16), dp(14))
                    background = android.graphics.drawable.GradientDrawable().apply {
                        shape = android.graphics.drawable.GradientDrawable.RECTANGLE
                        cornerRadius = dp(10).toFloat()
                        setColor(Color.parseColor("#1E1E1E"))
                    }

                    val params = LinearLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT)
                    params.bottomMargin = dp(8)
                    layoutParams = params
                }

                val icon = TextView(this).apply {
                    text = if (type == "DIR") "📁" else "📄"
                    textSize = 18f
                    setPadding(0, 0, dp(12), 0)
                }

                val label = TextView(this).apply {
                    text = name
                    textSize = 14f
                    setTextColor(Color.parseColor("#CCCCCC"))
                }

                item.addView(icon)
                item.addView(label)

                item.setOnClickListener {

                    if (type == "DIR") {
                        pathHistory.add(currentPath)
                        val next =
                            if (currentPath.isEmpty()) "$name\\"
                            else "$currentPath\\$name"
                        requestFolder(next)
                    } else {
                        ConnectionManager.send("FILE_REQ_DOWNLOAD|$currentPath\\$name")
                        Toast.makeText(this, "Downloading $name", Toast.LENGTH_SHORT).show()
                    }
                }

                listLayout.addView(item)
            }
        }
    }
}