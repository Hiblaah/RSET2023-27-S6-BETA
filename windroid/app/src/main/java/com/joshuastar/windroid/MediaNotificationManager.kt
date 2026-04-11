package com.joshuastar.windroid

import android.app.*
import android.content.*
import android.os.Build
import android.support.v4.media.MediaMetadataCompat
import android.support.v4.media.session.MediaSessionCompat
import android.support.v4.media.session.PlaybackStateCompat
import androidx.core.app.NotificationCompat
import androidx.media.app.NotificationCompat.MediaStyle
import org.json.JSONObject

// ─────────────────────────────────────────────
// Media Command Receiver
// ─────────────────────────────────────────────
class MediaCommandReceiver : BroadcastReceiver() {
    override fun onReceive(context: Context, intent: Intent) {
        val command = intent.getStringExtra("command") ?: return
        ConnectionManager.send(command)
    }
}

// ─────────────────────────────────────────────
// Media Notification Manager
// ─────────────────────────────────────────────
object MediaNotificationManager {

    private const val CHANNEL_ID = "media_control"
    private const val NOTIF_ID = 99

    var mediaSession: MediaSessionCompat? = null
    private var isPlaying = false
    private var title = ""
    private var artist = ""
    private var position = 0L
    private var duration = 0L

    fun init(context: Context) {
        createChannel(context)

        mediaSession = MediaSessionCompat(context, "windroid_media").apply {
            setCallback(object : MediaSessionCompat.Callback() {
                override fun onPlay() {
                    ConnectionManager.send("CMD:MEDIA_PLAY")
                }
                override fun onPause() {
                    ConnectionManager.send("CMD:MEDIA_PAUSE")
                }
                override fun onSkipToNext() {
                    ConnectionManager.send("CMD:MEDIA_NEXT")
                }
                override fun onSkipToPrevious() {
                    ConnectionManager.send("CMD:MEDIA_PREV")
                }
                override fun onSeekTo(pos: Long) {
                    ConnectionManager.send("CMD:MEDIA_SEEK:${pos / 1000}")
                }
            })
            isActive = true
        }
    }

    fun handle(context: Context, json: String) {
        try {
            val obj = JSONObject(json)

            if (!obj.optBoolean("active", false)) {
                cancel(context)
                return
            }

            title = obj.optString("title", "")
            artist = obj.optString("artist", "")
            isPlaying = obj.optBoolean("playing", false)
            position = obj.optLong("position", 0) * 1000
            duration = obj.optLong("duration", 0) * 1000

            updateSession()
            showNotification(context)

        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    private fun updateSession() {
        val session = mediaSession ?: return

        session.setMetadata(
            MediaMetadataCompat.Builder()
                .putString(MediaMetadataCompat.METADATA_KEY_TITLE, title)
                .putString(MediaMetadataCompat.METADATA_KEY_ARTIST, artist)
                .putLong(MediaMetadataCompat.METADATA_KEY_DURATION, duration)
                .build()
        )

        val state = if (isPlaying)
            PlaybackStateCompat.STATE_PLAYING
        else
            PlaybackStateCompat.STATE_PAUSED

        session.setPlaybackState(
            PlaybackStateCompat.Builder()
                .setState(state, position, 1f)
                .setActions(
                    PlaybackStateCompat.ACTION_PLAY or
                            PlaybackStateCompat.ACTION_PAUSE or
                            PlaybackStateCompat.ACTION_SKIP_TO_NEXT or
                            PlaybackStateCompat.ACTION_SKIP_TO_PREVIOUS or
                            PlaybackStateCompat.ACTION_SEEK_TO
                )
                .build()
        )
    }

    private fun showNotification(context: Context) {
        val session = mediaSession ?: return

        val prevAction = NotificationCompat.Action(
            android.R.drawable.ic_media_previous, "Previous",
            buildPendingIntent(context, "CMD:MEDIA_PREV")
        )

        val playPauseAction = NotificationCompat.Action(
            if (isPlaying) android.R.drawable.ic_media_pause
            else android.R.drawable.ic_media_play,
            if (isPlaying) "Pause" else "Play",
            buildPendingIntent(context,
                if (isPlaying) "CMD:MEDIA_PAUSE" else "CMD:MEDIA_PLAY")
        )

        val nextAction = NotificationCompat.Action(
            android.R.drawable.ic_media_next, "Next",
            buildPendingIntent(context, "CMD:MEDIA_NEXT")
        )

        val notification = NotificationCompat.Builder(context, CHANNEL_ID)
            .setContentTitle(title)
            .setContentText(artist)
            .setSmallIcon(R.mipmap.ic_launcher)
            .setVisibility(NotificationCompat.VISIBILITY_PUBLIC)
            .addAction(prevAction)
            .addAction(playPauseAction)
            .addAction(nextAction)
            .setStyle(
                MediaStyle()
                    .setMediaSession(session.sessionToken)
                    .setShowActionsInCompactView(0, 1, 2)
            )
            .setOngoing(isPlaying)
            .build()

        val manager = context.getSystemService(Context.NOTIFICATION_SERVICE)
                as NotificationManager
        manager.notify(NOTIF_ID, notification)
    }

    fun cancel(context: Context) {
        val manager = context.getSystemService(Context.NOTIFICATION_SERVICE)
                as NotificationManager
        manager.cancel(NOTIF_ID)
        mediaSession?.isActive = false
    }

    private fun buildPendingIntent(context: Context, command: String): PendingIntent {
        val intent = Intent("com.joshuastar.windroid.MEDIA_CMD").apply {
            putExtra("command", command)
            setPackage(context.packageName)
        }
        return PendingIntent.getBroadcast(
            context, command.hashCode(), intent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )
    }

    private fun createChannel(context: Context) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                CHANNEL_ID,
                "Media Controls",
                NotificationManager.IMPORTANCE_LOW
            )
            val manager = context.getSystemService(Context.NOTIFICATION_SERVICE)
                    as NotificationManager
            manager.createNotificationChannel(channel)
        }
    }
}