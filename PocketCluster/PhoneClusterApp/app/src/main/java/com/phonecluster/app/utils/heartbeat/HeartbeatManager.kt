package com.phonecluster.app.utils.heartbeat

import android.content.Context
import android.util.Log
import com.phonecluster.app.utils.DeviceInfoProvider
import com.phonecluster.app.utils.websocket.WebSocketManager
import kotlinx.coroutines.*

object HeartbeatManager {

    private const val HEARTBEAT_INTERVAL_MS = 10_000L
    private var job: Job? = null

    private lateinit var appContext: Context

    fun start(
        context: Context,
        serverBaseUrl: String,
        serverIp: String,
        deviceId: Int,
        mode:String
    ) {
        if (job != null) return

        appContext = context.applicationContext

        job = CoroutineScope(Dispatchers.IO).launch {
            while (isActive) {

                try {
                    val availableStorage =
                        DeviceInfoProvider.getAvailableStorageBytes()

                    HeartbeatHttpClient.sendHeartbeat(
                        serverBaseUrl = serverBaseUrl,
                        deviceId = deviceId,
                        availableStorage = availableStorage,
                        mode = mode
                    )
                } catch (e: Exception) {
                    Log.d("HEARTBEAT", "Heartbeat failed: ${e.message}")
                }

                // 🔴 Recover WS if dead
                if (!WebSocketManager.isConnected()) {
                    Log.d("WS_DEBUG", "Heartbeat detected missing WS. Reconnecting.")
                    WebSocketManager.connect(appContext, serverIp)
                }

                delay(HEARTBEAT_INTERVAL_MS)
            }
        }
    }

    fun stop() {
        job?.cancel()
        job = null
    }
}