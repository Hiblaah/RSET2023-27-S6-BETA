package com.emergency.classification;

import android.Manifest;
import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.content.pm.PackageManager;
import android.os.Build;
import android.os.Bundle;
import android.util.Log;
import android.widget.Button;
import android.widget.TextView;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.ContextCompat;
import androidx.localbroadcastmanager.content.LocalBroadcastManager;

public class MainActivity extends AppCompatActivity {
    private static final String TAG = "MainActivity";

    private TextView statusText;
    private TextView classText;
    private TextView confidenceText;
    private TextView dbText;
    private Button   startStopButton;

    private boolean isServiceRunning = false;

    // ── Permission launcher ───────────────────────────────────────────────────
    private final ActivityResultLauncher<String[]> permissionLauncher =
            registerForActivityResult(
                    new ActivityResultContracts.RequestMultiplePermissions(),
                    result -> {
                        boolean audioGranted = Boolean.TRUE.equals(
                                result.get(Manifest.permission.RECORD_AUDIO));
                        if (audioGranted) {
                            startDetectionService();
                        } else {
                            statusText.setText("❌ Microphone permission denied");
                        }
                    });

    // ── Detection result receiver ─────────────────────────────────────────────
    private final BroadcastReceiver detectionReceiver = new BroadcastReceiver() {
        @Override
        public void onReceive(Context context, Intent intent) {
            String  className   = intent.getStringExtra(DetectionService.EXTRA_CLASS_NAME);
            float   confidence  = intent.getFloatExtra(DetectionService.EXTRA_CONFIDENCE, 0f);
            boolean isEmergency = intent.getBooleanExtra(DetectionService.EXTRA_IS_EMERGENCY, false);
            boolean isDbAlert   = intent.getBooleanExtra(DetectionService.EXTRA_IS_DB_ALERT, false);

            if (className == null) return;

            Log.d(TAG, "Received: " + className + " " + (int)(confidence * 100) + "%");

            if (isDbAlert) {
                classText.setText("⚠️ Loud Noise");
                confidenceText.setText("Dangerous sound level");
                statusText.setText("dB threshold exceeded");
            } else if (isEmergency) {
                classText.setText(emojiFor(className) + " " + className);
                confidenceText.setText("Confidence: " + (int)(confidence * 100) + "%");
                statusText.setText("🚨 EMERGENCY DETECTED");
            } else {
                classText.setText(className);
                confidenceText.setText((int)(confidence * 100) + "%");
                statusText.setText("Monitoring...");
            }
        }
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        statusText     = findViewById(R.id.statusText);
        classText      = findViewById(R.id.classText);
        confidenceText = findViewById(R.id.confidenceText);
        dbText         = findViewById(R.id.dbText);
        startStopButton = findViewById(R.id.startStopButton);

        startStopButton.setOnClickListener(v -> {
            if (isServiceRunning) {
                stopDetectionService();
            } else {
                checkPermissionsAndStart();
            }
        });

        // Register for detection results
        LocalBroadcastManager.getInstance(this)
                .registerReceiver(detectionReceiver,
                        new IntentFilter(DetectionService.ACTION_DETECTION_RESULT));
    }

    private void checkPermissionsAndStart() {
        // Check microphone permission
        boolean audioOk = ContextCompat.checkSelfPermission(this,
                Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED;

        // Check notification permission on Android 13+
        boolean notifOk = true;
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            notifOk = ContextCompat.checkSelfPermission(this,
                    Manifest.permission.POST_NOTIFICATIONS) == PackageManager.PERMISSION_GRANTED;
        }

        if (audioOk && notifOk) {
            startDetectionService();
        } else {
            // Request all needed permissions at once
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                permissionLauncher.launch(new String[]{
                        Manifest.permission.RECORD_AUDIO,
                        Manifest.permission.POST_NOTIFICATIONS
                });
            } else {
                permissionLauncher.launch(new String[]{
                        Manifest.permission.RECORD_AUDIO
                });
            }
        }
    }

    private void startDetectionService() {
        Intent serviceIntent = new Intent(this, DetectionService.class);
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            startForegroundService(serviceIntent);
        } else {
            startService(serviceIntent);
        }
        isServiceRunning = true;
        startStopButton.setText("Stop Monitoring");
        statusText.setText("Monitoring...");
        classText.setText("—");
        confidenceText.setText("—");
        Log.d(TAG, "Detection service started");
    }

    private void stopDetectionService() {
        stopService(new Intent(this, DetectionService.class));
        isServiceRunning = false;
        startStopButton.setText("Start Monitoring");
        statusText.setText("Stopped");
        classText.setText("—");
        confidenceText.setText("—");
        Log.d(TAG, "Detection service stopped");
    }

    private String emojiFor(String cls) {
        switch (cls) {
            case "Glass Breaking": return "💥";
            case "Siren":          return "🚨";
            case "Baby Crying":    return "👶";
            case "Car Horn":       return "📯";
            case "Explosion":      return "🔥";
            default:               return "⚠️";
        }
    }

    @Override
    protected void onDestroy() {
        LocalBroadcastManager.getInstance(this)
                .unregisterReceiver(detectionReceiver);
        super.onDestroy();
    }
}