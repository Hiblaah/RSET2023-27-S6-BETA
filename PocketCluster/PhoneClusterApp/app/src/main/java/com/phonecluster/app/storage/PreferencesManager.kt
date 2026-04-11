package com.phonecluster.app.storage

import android.content.Context
import android.content.SharedPreferences

// what is a preference file
//A small XML file stored on the phone’s internal storage, private to your app, that Android manages for you.

/*
preferencesmanager.kt stores important small values like the device_id
so the app can remember who it is(device_id) when it starts again,
instead of registering as a new device every time
 */

object PreferencesManager {

    private const val PREF_NAME = "phone_cluster_prefs"
    private const val KEY_DEVICE_ID = "device_id"
    private const val KEY_IS_REGISTERED = "is_registered"

    private fun getPreferences(context: Context): SharedPreferences {
        return context.getSharedPreferences(
            PREF_NAME,
            Context.MODE_PRIVATE
        )
    }

    fun saveDeviceId(context: Context, deviceId: Int) {
        getPreferences(context)
            .edit()
            .putInt(KEY_DEVICE_ID, deviceId)
            .apply()
    }

    fun getDeviceId(context: Context): Int? {
        val prefs = getPreferences(context)
        return if (prefs.contains(KEY_DEVICE_ID)) {
            prefs.getInt(KEY_DEVICE_ID, -1)
        } else {
            null
        }
    }

    fun setRegistered(context: Context, registered: Boolean) {
        getPreferences(context)
            .edit()
            .putBoolean(KEY_IS_REGISTERED, registered)
            .apply()
    }

    fun isRegistered(context: Context): Boolean {
        return getPreferences(context)
            .getBoolean(KEY_IS_REGISTERED, false)
    }

    fun clear(context: Context) {
        getPreferences(context)
            .edit()
            .clear()
            .apply()
    }
}
