package com.phonecluster.app.utils.websocket

import android.util.Log
import okhttp3.*
import okio.ByteString
import org.json.JSONObject
import java.util.concurrent.TimeUnit

class DeviceWebSocketClient(
    private val serverWsUrl: String,
    private val onOpenCallback: () -> Unit,
    private val onMessageReceived: (JSONObject) -> Unit,
    private val onDisconnected: () -> Unit
) {

    private var webSocket: WebSocket? = null

    private val client = OkHttpClient.Builder()
        .pingInterval(15, TimeUnit.SECONDS)
        .build()

    fun connect(registerPayload: JSONObject) {
        Log.d("WS_DEBUG", "Attempting connection to $serverWsUrl")

        val request = Request.Builder()
            .url(serverWsUrl)
            .build()

        webSocket = client.newWebSocket(request, object : WebSocketListener() {

            override fun onOpen(ws: WebSocket, response: Response) {
                Log.d("WS_DEBUG", "WebSocket OPENED")
                ws.send(registerPayload.toString())
                onOpenCallback()
            }

            override fun onMessage(ws: WebSocket, text: String) {
                Log.d("WS_DEBUG", "Message received: $text")
                val json = JSONObject(text)
                onMessageReceived(json)
            }

            override fun onMessage(ws: WebSocket, bytes: ByteString) {
                Log.d("WS_DEBUG", "Binary message received")
            }

            override fun onFailure(ws: WebSocket, t: Throwable, response: Response?) {
                Log.e("WS_DEBUG", "WebSocket FAILURE: ${t.message}")
                response?.let {
                    Log.e("WS_DEBUG", "HTTP response code: ${it.code}")
                }
                onDisconnected()
            }

            override fun onClosed(ws: WebSocket, code: Int, reason: String) {
                Log.d("WS_DEBUG", "WebSocket CLOSED: $code / $reason")
                onDisconnected()
            }
        })
    }

    fun send(json: JSONObject) {
        Log.d("WS_DEBUG", "Sending: $json")
        webSocket?.send(json.toString())
    }

    fun disconnect() {
        Log.d("WS_DEBUG", "Closing WebSocket")
        webSocket?.close(1000, "Client disconnect")
        webSocket = null
    }
}