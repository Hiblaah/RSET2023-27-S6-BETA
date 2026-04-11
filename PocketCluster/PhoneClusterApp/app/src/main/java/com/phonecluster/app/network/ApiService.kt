package com.phonecluster.app.network

import retrofit2.http.Body
import retrofit2.http.POST

// Request body sent to backend
data class DeviceRegistrationRequest(
    val user_id: Int,
    val device_name: String,
    val fingerprint: String,
    val storage_capacity: Long,
    val available_storage: Long
)

// Response body returned by backend
data class DeviceRegistrationResponse(
    val device_id: Int,
    val status: String   //"registered" or "already_registered"
)

// API interface
interface ApiService {

    @POST("/devices/register")
    suspend fun registerDevice(
        @Body request: DeviceRegistrationRequest
    ): DeviceRegistrationResponse
}
