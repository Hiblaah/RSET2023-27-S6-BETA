package com.joshuastar.windroid

import android.content.Intent
import android.graphics.Color
import android.os.Bundle
import android.view.Gravity
import android.view.ViewGroup
import android.widget.*
import androidx.activity.ComponentActivity
import java.net.DatagramPacket
import java.net.DatagramSocket

const val DISCOVERY_PORT = 9876
const val DISCOVERY_MESSAGE = "WINDROID_SERVER"
const val PREF_NAME = "windroid_prefs"
const val PREF_SERVER_IP = "server_ip"
const val CLIPBOARD_PORT = 1234

class DiscoveryActivity : ComponentActivity() {

    private lateinit var serverListLayout: LinearLayout
    private lateinit var statusText: TextView
    private val discoveredServers = mutableSetOf<String>()
    private var isListening = true
    private var discoverySocket: DatagramSocket? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val prefs = getSharedPreferences(PREF_NAME, MODE_PRIVATE)
        val savedIp = prefs.getString(PREF_SERVER_IP, null)

        if (savedIp != null) {
            ConnectionManager.PersistentService.start(applicationContext)
            goToMain(savedIp)
            return
        }

        createUI()
        startListening()
    }

    private fun createUI() {
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(Color.parseColor("#1E1E1E"))
            setPadding(40, 80, 40, 40)
            gravity = Gravity.TOP or Gravity.CENTER_HORIZONTAL
        }

        val title = TextView(this).apply {
            text = "Windroid"
            textSize = 32f
            setTextColor(Color.WHITE)
            gravity = Gravity.CENTER
            setPadding(0, 0, 0, 8)
        }

        val subtitle = TextView(this).apply {
            text = "Searching for PC on your network..."
            textSize = 14f
            setTextColor(Color.parseColor("#888888"))
            gravity = Gravity.CENTER
            setPadding(0, 0, 0, 60)
        }

        statusText = subtitle

        val spinner = ProgressBar(this).apply {
            layoutParams = LinearLayout.LayoutParams(80, 80).apply {
                gravity = Gravity.CENTER_HORIZONTAL
                bottomMargin = 40
            }
        }

        val listLabel = TextView(this).apply {
            text = "Available Servers"
            textSize = 13f
            setTextColor(Color.parseColor("#666666"))
            setPadding(0, 0, 0, 16)
        }

        serverListLayout = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            layoutParams = LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
            )
        }

        val manualBtn = TextView(this).apply {
            text = "Enter IP manually"
            textSize = 13f
            setTextColor(Color.parseColor("#555555"))
            gravity = Gravity.CENTER
            setPadding(0, 60, 0, 0)
            setOnClickListener { showManualIPDialog() }
        }

        root.addView(title)
        root.addView(subtitle)
        root.addView(spinner)
        root.addView(listLabel)
        root.addView(serverListLayout)
        root.addView(manualBtn)

        setContentView(root)
    }

    private fun startListening() {
        discoveredServers.clear()
        isListening = true
        Thread {
            try {
                discoverySocket?.close()
                val socket = DatagramSocket(null).apply {
                    reuseAddress = true
                    bind(java.net.InetSocketAddress(DISCOVERY_PORT))
                    broadcast = true
                }
                discoverySocket = socket
                val buffer = ByteArray(256)

                while (isListening) {
                    val packet = DatagramPacket(buffer, buffer.size)
                    socket.receive(packet)

                    val message = String(packet.data, 0, packet.length)
                    val senderIp = packet.address.hostAddress ?: continue

                    if (message == DISCOVERY_MESSAGE && !discoveredServers.contains(senderIp)) {
                        discoveredServers.add(senderIp)
                        runOnUiThread { addServerToList(senderIp) }
                    }
                }

                socket.close()
            } catch (e: Exception) {
                e.printStackTrace()
            }
        }.start()
    }

    private fun addServerToList(ip: String) {
        statusText.text = "Found ${discoveredServers.size} server(s)"

        val btn = TextView(this).apply {
            text = "🖥️  Windroid PC  —  $ip"
            textSize = 16f
            setTextColor(Color.WHITE)
            setBackgroundColor(Color.parseColor("#2D2D2D"))
            setPadding(30, 30, 30, 30)
            gravity = Gravity.CENTER_VERTICAL
            layoutParams = LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
            ).apply { bottomMargin = 12 }

            setOnClickListener {
                getSharedPreferences(PREF_NAME, MODE_PRIVATE)
                    .edit()
                    .putString(PREF_SERVER_IP, ip)
                    .apply()
                ConnectionManager.PersistentService.start(applicationContext)
                goToMain(ip)
            }
        }

        serverListLayout.addView(btn)
    }

    private fun showManualIPDialog() {
        val input = EditText(this).apply {
            hint = "192.168.x.x"
            setTextColor(Color.WHITE)
            setHintTextColor(Color.parseColor("#555555"))
            setBackgroundColor(Color.parseColor("#2D2D2D"))
            setPadding(20, 20, 20, 20)
        }

        android.app.AlertDialog.Builder(this)
            .setTitle("Enter Server IP")
            .setView(input)
            .setPositiveButton("Connect") { _, _ ->
                val ip = input.text.toString().trim()
                if (ip.isNotBlank()) {
                    getSharedPreferences(PREF_NAME, MODE_PRIVATE)
                        .edit()
                        .putString(PREF_SERVER_IP, ip)
                        .apply()
                    ConnectionManager.PersistentService.start(applicationContext)
                    goToMain(ip)
                }
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    private fun goToMain(ip: String) {
        isListening = false
        val intent = Intent(this, MainActivity::class.java).apply {
            putExtra("server_ip", ip)
        }
        startActivity(intent)
        finish()
    }

    override fun onDestroy() {
        super.onDestroy()
        isListening = false
        discoverySocket?.close()
        discoverySocket = null
    }
}