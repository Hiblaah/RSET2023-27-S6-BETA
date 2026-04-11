package com.emergency.classification;

import android.app.Application;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.os.Build;

public class EmergencyApp extends Application {

    public static final String MONITOR_CHANNEL_ID = "MonitorChannel";
    public static final String ALERT_CHANNEL_ID   = "AlertChannel";
    public static final String DB_CHANNEL_ID      = "DbAlertChannel";

    @Override
    public void onCreate() {
        super.onCreate();
        createNotificationChannels();
    }

    private void createNotificationChannels() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            NotificationManager nm = getSystemService(NotificationManager.class);

            nm.createNotificationChannel(new NotificationChannel(
                    MONITOR_CHANNEL_ID, "Sound Monitor",
                    NotificationManager.IMPORTANCE_LOW));

            NotificationChannel alert = new NotificationChannel(
                    ALERT_CHANNEL_ID, "Emergency Alerts",
                    NotificationManager.IMPORTANCE_HIGH);
            alert.enableVibration(true);
            alert.setVibrationPattern(new long[]{0, 400, 200, 400});
            nm.createNotificationChannel(alert);

            NotificationChannel db = new NotificationChannel(
                    DB_CHANNEL_ID, "Loud Noise Warnings",
                    NotificationManager.IMPORTANCE_HIGH);
            db.enableVibration(true);
            db.setVibrationPattern(new long[]{0, 300, 150, 300});
            nm.createNotificationChannel(db);
        }
    }
}