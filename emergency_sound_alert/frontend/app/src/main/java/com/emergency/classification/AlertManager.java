package com.emergency.classification;

import android.content.Context;
import android.os.Build;
import android.os.VibrationEffect;
import android.os.Vibrator;
import android.os.VibratorManager;
import android.util.Log;

public class AlertManager {
    private static final String TAG = "AlertManager";

    private final Context  context;
    private final Vibrator vibrator;

    // Vibration patterns (ms): wait, vibrate, wait, vibrate...
    private static final long[] EMERGENCY_PATTERN = {0, 500, 200, 500, 200, 500};
    private static final long[] DB_PATTERN        = {0, 300, 150, 300};

    public AlertManager(Context context) {
        this.context = context;

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            VibratorManager vm = (VibratorManager) context
                    .getSystemService(Context.VIBRATOR_MANAGER_SERVICE);
            vibrator = (vm != null) ? vm.getDefaultVibrator() : null;
        } else {
            vibrator = (Vibrator) context.getSystemService(Context.VIBRATOR_SERVICE);
        }

        Log.d(TAG, "AlertManager initialized");
    }

    /**
     * Trigger full emergency alert — vibration only.
     * Notification and popup are handled by DetectionService.
     * className is passed in case future versions need class-specific behaviour.
     */
    public void triggerAlert(String className, Object unused) {
        Log.d(TAG, "triggerAlert: " + className);
        vibrate(EMERGENCY_PATTERN,0);
    }

    /**
     * Trigger dB threshold alert — shorter vibration pattern.
     */
    public void triggerDbAlert() {
        Log.d(TAG, "triggerDbAlert");
        vibrate(DB_PATTERN);
    }

    public void stopVibration() {
        if (vibrator != null) {
            vibrator.cancel();
            Log.d(TAG, "Vibration cancelled");
        }
    }
    private void vibrate(long[] pattern, int repeat) {
        if (vibrator == null || !vibrator.hasVibrator()) {
            Log.w(TAG, "No vibrator available");
            return;
        }
        try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                vibrator.vibrate(VibrationEffect.createWaveform(pattern, repeat));
            } else {
                vibrator.vibrate(pattern, repeat);
            }
        } catch (Exception e) {
            Log.e(TAG, "Vibration error", e);
        }
    }
    private void vibrate(long[] pattern) {
        if (vibrator == null || !vibrator.hasVibrator()) {
            Log.w(TAG, "No vibrator available");
            return;
        }
        try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                vibrator.vibrate(VibrationEffect.createWaveform(pattern, -1));
            } else {
                vibrator.vibrate(pattern, -1);
            }
        } catch (Exception e) {
            Log.e(TAG, "Vibration error", e);
        }
    }
}