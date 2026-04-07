package com.joshuastar.windroid

import android.bluetooth.BluetoothAdapter
import android.content.*
import android.media.AudioManager
import android.os.BatteryManager

object PhoneStatusSender {

    private var started = false

    fun start(context: Context) {

        if (started) return
        started = true

        val filter = IntentFilter(Intent.ACTION_BATTERY_CHANGED)

        context.registerReceiver(object : BroadcastReceiver() {

            override fun onReceive(ctx: Context, intent: Intent) {

                val level = intent.getIntExtra(BatteryManager.EXTRA_LEVEL, -1)
                val scale = intent.getIntExtra(BatteryManager.EXTRA_SCALE, -1)

                val batteryPercent =
                    if (level >= 0 && scale > 0)
                        (level * 100) / scale
                    else -1

                val status = intent.getIntExtra(BatteryManager.EXTRA_STATUS, -1)
                val isCharging =
                    status == BatteryManager.BATTERY_STATUS_CHARGING ||
                            status == BatteryManager.BATTERY_STATUS_FULL

                val audioManager =
                    context.getSystemService(Context.AUDIO_SERVICE) as AudioManager

                val mode = when (audioManager.ringerMode) {

                    AudioManager.RINGER_MODE_SILENT -> "SILENT"

                    AudioManager.RINGER_MODE_VIBRATE -> "VIBRATE"

                    else -> "NORMAL"
                }

                val btAdapter = BluetoothAdapter.getDefaultAdapter()

                val btEnabled =
                    btAdapter != null && btAdapter.isEnabled

                if (ConnectionManager.isConnected()) {

                    ConnectionManager.send(
                        "STATUS|" +
                                "BATTERY:$batteryPercent|" +
                                "MODE:$mode|" +
                                "CHARGING:$isCharging|" +
                                "BT:$btEnabled"
                    )
                }
            }
        }, filter)
    }
}