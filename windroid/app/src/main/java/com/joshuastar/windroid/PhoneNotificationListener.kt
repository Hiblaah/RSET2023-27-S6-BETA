package com.joshuastar.windroid

import android.app.Person
import android.graphics.*
import android.graphics.drawable.BitmapDrawable
import android.graphics.drawable.Drawable
import android.os.Bundle
import android.service.notification.NotificationListenerService
import android.service.notification.StatusBarNotification
import android.util.Base64
import android.util.Log
import java.io.ByteArrayOutputStream

class PhoneNotificationListener : NotificationListenerService() {

    companion object {
        private const val TAG = "NotifListener"

        private val IGNORED_PACKAGES = setOf(
            "android",
            "com.android.systemui",
            "com.android.phone",
            "com.google.android.gms",
            "com.joshuastar.windroid"
        )
    }

    override fun onNotificationPosted(sbn: StatusBarNotification) {
        val packageName = sbn.packageName
        if (packageName in IGNORED_PACKAGES) return
        if (sbn.isOngoing) return
        if (sbn.notification.flags and android.app.Notification.FLAG_GROUP_SUMMARY != 0) return

        val key    = sbn.key ?: return
        val extras = sbn.notification.extras

        val title = extras.getCharSequence("android.title")?.toString() ?: ""
        val text  = extras.getCharSequence("android.text")?.toString()  ?: ""

        // ── Extract sender name + icon ───────────────────────────────────────
        var senderName       = ""
        var senderIconBase64 = ""

        try {
            val messages = extras.getParcelableArray("android.messages")
            if (!messages.isNullOrEmpty()) {
                val lastMsg = messages.last() as Bundle

                senderName = lastMsg.getCharSequence("sender")?.toString()
                    ?: lastMsg.getString("sender")
                            ?: ""

                if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.P) {
                    val person   = lastMsg.getParcelable<Person>("sender_person")
                    val drawable = person?.icon?.loadDrawable(this)
                    if (drawable != null) {
                        senderIconBase64 = bitmapToBase64(drawableToBitmap(drawable))
                    }
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "MessagingStyle parse error: ${e.message}")
        }

        if (title.isBlank() && text.isBlank()) return

        val appName = getAppName(packageName)

        // ── Icon: sender photo → app icon ────────────────────────────────────
        val iconBase64 = if (senderIconBase64.isNotEmpty()) senderIconBase64
        else try {
            bitmapToBase64(drawableToBitmap(packageManager.getApplicationIcon(packageName)))
        } catch (e: Exception) { "" }

        // ── Sanitize ─────────────────────────────────────────────────────────
        val safeAppName = appName.sanitize()
        val safeTitle   = title.sanitize()       // group name or DM name
        val safeSender  = senderName.sanitize()  // individual sender
        val safeText    = text.sanitize()
        val safeKey     = key.replace("|", "_").replace("\n", "").replace("\r", "")

        // Format: NOTIF|appName|chatTitle|senderName|text|packageName|key|iconBase64
        val message = "NOTIF|$safeAppName|$safeTitle|$safeSender|$safeText|$packageName|$safeKey|$iconBase64"
        ConnectionManager.send(message)

        Log.d(TAG, "[$safeAppName] chat='$safeTitle' sender='$safeSender' text='$safeText'")
    }

    override fun onNotificationRemoved(sbn: StatusBarNotification) {
        Log.d(TAG, "Notification removed: ${sbn.key}")
    }

    private fun String.sanitize() =
        replace("|", "｜").replace("\n", " ").replace("\r", "")

    private fun getAppName(packageName: String): String {
        return try {
            val info = packageManager.getApplicationInfo(packageName, 0)
            packageManager.getApplicationLabel(info).toString()
        } catch (e: Exception) { packageName }
    }

    private fun drawableToBitmap(drawable: Drawable): Bitmap {
        if (drawable is BitmapDrawable && drawable.bitmap != null) return drawable.bitmap
        val w = if (drawable.intrinsicWidth  > 0) drawable.intrinsicWidth  else 128
        val h = if (drawable.intrinsicHeight > 0) drawable.intrinsicHeight else 128
        val bitmap = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(bitmap)
        drawable.setBounds(0, 0, w, h)
        drawable.draw(canvas)
        return bitmap
    }

    private fun bitmapToBase64(bitmap: Bitmap): String {
        val scaled = Bitmap.createScaledBitmap(bitmap, 64, 64, true)
        val stream = ByteArrayOutputStream()
        scaled.compress(Bitmap.CompressFormat.PNG, 100, stream)
        return Base64.encodeToString(stream.toByteArray(), Base64.NO_WRAP)
            .replace("\n", "").replace("\r", "")
    }
}