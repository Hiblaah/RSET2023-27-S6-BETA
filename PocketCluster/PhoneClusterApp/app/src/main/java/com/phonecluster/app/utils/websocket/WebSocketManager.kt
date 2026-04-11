package com.phonecluster.app.utils.websocket

import android.content.Context
import android.util.Log
import com.phonecluster.app.storage.PreferencesManager
import com.phonecluster.app.storage.ChunkStorage
import com.phonecluster.app.utils.DeviceInfoProvider
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject
import java.security.MessageDigest

object WebSocketManager {

    private var wsClient: DeviceWebSocketClient? = null
    private var isConnected = false
    private lateinit var appContext: Context

    fun connect(context: Context, serverIp: String) {
        if (isConnected) {
            Log.d("WS_DEBUG", "Already connected, skipping")
            return
        }
//        if (isConnected) return

        val deviceId = PreferencesManager.getDeviceId(context) ?: return
        appContext = context.applicationContext

        val registerPayload = JSONObject().apply {
            put("type", "register")
            put("device_id", deviceId)
            put("fingerprint", DeviceInfoProvider.getDeviceFingerprint(context))
            put("device_name", DeviceInfoProvider.getDeviceName())
            put("storage_capacity", DeviceInfoProvider.getTotalStorageBytes())
            put("available_storage", DeviceInfoProvider.getAvailableStorageBytes())
        }

        val wsUrl = "ws://$serverIp:8000/ws/device"

        wsClient = DeviceWebSocketClient(
            serverWsUrl = wsUrl,

            onOpenCallback = {
                Log.d("WS_DEBUG", "Connection established")
                isConnected = true
            },

            onMessageReceived = { msg ->
                handleServerMessage(msg)
            },

            onDisconnected = {
                Log.d("WS_DEBUG", "Disconnected from server")
                isConnected = false

                // Attempt reconnect after delay
                CoroutineScope(Dispatchers.IO).launch {
                    delay(3000)
                    connect(appContext, serverIp)
                }
            }
        )

        wsClient?.connect(registerPayload)
        isConnected = true
    }

    fun disconnect() {
        wsClient?.disconnect()
        wsClient = null
        isConnected = false
    }
    fun isConnected(): Boolean {
        return isConnected
    }

    private fun handleServerMessage(msg: JSONObject) {
        Log.d("WS_DEBUG", "Handling message: $msg")

        when (msg.optString("type")) {

            "ready" -> {
                Log.d("WS_DEBUG", "Connected to server")
            }

            "command" -> {
                val command = msg.optString("command")
                val data = msg.optJSONObject("data")

                if (data == null) {
                    Log.e("WS_DEBUG", "Command without data: $msg")
                    return
                }

                when (command) {
                    "DOWNLOAD_CHUNK" -> handleDownloadChunk(data)
                    "PUSH_CHUNK" -> handlePushChunk(data)

                    else -> {
                        Log.d("WS_DEBUG", "Unknown command: $command")
                    }
                }
            }

            else -> {
                Log.d("WS_DEBUG", "Unknown message type: $msg")
            }
        }
    }
    private fun handleDownloadChunk(msg: JSONObject) {
        val chunkId = msg.getLong("chunk_id")
        val downloadUrl = msg.getString("download_url")
        val expectedHash = msg.getString("expected_hash")

        Log.d("WS", "DOWNLOAD_CHUNK received for $chunkId")
        Log.d("WS", "Downloading from $downloadUrl")

        CoroutineScope(Dispatchers.IO).launch {
            try {
                val httpClient = OkHttpClient()

                val request = Request.Builder()
                    .url(downloadUrl)
                    .get()
                    .build()

                val response = httpClient.newCall(request).execute()

                if (!response.isSuccessful) {
                    throw RuntimeException("Download failed: ${response.code}")
                }

                val bodyBytes = response.body?.bytes()
                    ?: throw RuntimeException("Empty response body")

                // 1️⃣ Validate SHA256
                val actualHash = sha256Hex(bodyBytes)
                if (actualHash != expectedHash) {
                    throw RuntimeException("Hash mismatch for chunk $chunkId")
                }

                Log.d("WS", "Hash verified for chunk $chunkId")

                // 2️⃣ Store locally
                ChunkStorage.writeChunk(
                    context = appContext,
                    chunkId = chunkId,
                    data = bodyBytes
                )

                Log.d("WS", "Chunk $chunkId stored locally")

                // 3️⃣ Send ACK to server
                wsClient?.send(
                    JSONObject()
                        .put("type", "CHUNK_STORED_SUCCESS")
                        .put("chunk_id", chunkId)
                )

                Log.d("WS", "CHUNK_STORED_SUCCESS sent for $chunkId")

            } catch (e: Exception) {
                Log.e("WS", "DOWNLOAD_CHUNK failed: ${e.message}")

                wsClient?.send(
                    JSONObject()
                        .put("type", "CHUNK_STORE_FAILED")
                        .put("chunk_id", chunkId)
                        .put("error", e.message ?: "unknown")
                )
            }
        }
    }

    private fun handlePushChunk(data: JSONObject) {
        val chunkId = data.getLong("chunk_id")
        val targetUrl = data.getString("target_url")

        CoroutineScope(Dispatchers.IO).launch {
            try {
                val chunkBytes = ChunkStorage.readChunk(appContext, chunkId)

                val requestBody = chunkBytes.toRequestBody()

                val request = Request.Builder()
                    .url(targetUrl)
                    .post(requestBody)
                    .build()

                OkHttpClient().newCall(request).execute()

                Log.d("WS", "PUSH_CHUNK uploaded $chunkId")

            } catch (e: Exception) {
                Log.e("WS", "PUSH_CHUNK failed: ${e.message}")
            }
        }
    }

    private fun sha256Hex(data: ByteArray): String {
        val digest = MessageDigest.getInstance("SHA-256")
        val hash = digest.digest(data)
        return hash.joinToString("") { "%02x".format(it) }
    }
}