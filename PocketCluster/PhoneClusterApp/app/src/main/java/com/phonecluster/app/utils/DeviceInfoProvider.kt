package com.phonecluster.app.utils

import android.content.Context
import android.os.Build
import android.os.Environment
import android.os.StatFs
import android.provider.Settings

object DeviceInfoProvider {

    //returns the stable unique fingerprint
    fun getDeviceFingerprint(context: Context): String {
        return Settings.Secure.getString(
            context.contentResolver, //system gateway that lets apps read/write device info
            Settings.Secure.ANDROID_ID
        )
    }
    //returns the readable device name
    //eg Samsung Galaxy M14
    fun getDeviceName(): String {
        val manufacturer = Build.MANUFACTURER
        val model = Build.MODEL

        return if (model.startsWith(manufacturer, ignoreCase = true)) {
            model.replaceFirstChar { it.uppercase() }
        } else {
            "${manufacturer.replaceFirstChar { it.uppercase() }} $model"
        }
    }
    //returns total internal storage capacity in bytes
    //TODO SHOULD IT BE IN BYTES?
    fun getTotalStorageBytes(): Long {
        val statFs = StatFs(Environment.getDataDirectory().path)
        return statFs.blockSizeLong * statFs.blockCountLong
    }

    //returns available internal storage capacity in bytes
    fun getAvailableStorageBytes(): Long {
        val statFs = StatFs(Environment.getDataDirectory().path)
        return statFs.blockSizeLong * statFs.availableBlocksLong
    }
}