package com.phonecluster.app.utils.heartbeat

import android.util.Log
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject
import java.io.IOException
import java.util.concurrent.TimeUnit

object HeartbeatHttpClient {

    private const val JSON = "application/json"

    private val client = OkHttpClient.Builder()
        .connectTimeout(5, TimeUnit.SECONDS)
        .readTimeout(5, TimeUnit.SECONDS)
        .build()

    fun sendHeartbeat(
        serverBaseUrl: String,
        deviceId: Int,
        availableStorage: Long?,
        mode: String
    ){
        val payload = JSONObject().apply {
            put("device_id", deviceId)
            put("mode", mode)

            availableStorage?.let {
                put("available_storage", it)
            }
        }

        val body = payload.toString()
            .toRequestBody(JSON.toMediaType())

        val request = Request.Builder()
            .url("$serverBaseUrl/devices/heartbeat")
            .post(body)
            .build()

        client.newCall(request).enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) {
                Log.e("Heartbeat", "Failed", e)
            }

            override fun onResponse(call: Call, response: Response) {
                Log.d("Heartbeat", "code=${response.code}")
                response.body?.string()?.let { Log.d("Heartbeat", it) }
                response.close()
            }
        })
    }
}
