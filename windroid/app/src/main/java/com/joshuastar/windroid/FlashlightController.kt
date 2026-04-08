package com.joshuastar.windroid

import android.content.Context
import android.hardware.camera2.CameraManager

object FlashlightController {

    private var cameraManager: CameraManager? = null
    private var cameraId: String? = null
    private var initialized = false

    fun init(context: Context) {

        if (initialized) return
        initialized = true

        cameraManager =
            context.getSystemService(Context.CAMERA_SERVICE) as CameraManager

        cameraId = cameraManager?.cameraIdList?.firstOrNull()
    }

    fun set(on: Boolean) {

        try {
            cameraManager?.setTorchMode(cameraId!!, on)
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }
}