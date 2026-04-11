package com.emergency.classification;

import android.app.KeyguardManager;
import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.graphics.Color;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.TextView;
import androidx.appcompat.app.AppCompatActivity;
import androidx.localbroadcastmanager.content.LocalBroadcastManager;

public class AlertPopupActivity extends AppCompatActivity {
    private static final String TAG       = "AlertPopupActivity";
    private static final int    AUTO_DISMISS_MS = 15000; // auto dismiss after 15 seconds

    private Handler handler;
    private Runnable autoDismissRunnable;

    // Flash animation colors per class
    private static final int COLOR_GLASS     = 0xFF1565C0; // dark blue
    private static final int COLOR_SIREN     = 0xFFB71C1C; // dark red
    private static final int COLOR_BABY      = 0xFF6A1B9A; // purple
    private static final int COLOR_HORN      = 0xFFE65100; // deep orange
    private static final int COLOR_EXPLOSION = 0xFFBF360C; // deep red-orange
    private static final int COLOR_LOUD      = 0xFF37474F; // dark grey
    private static final int COLOR_DEFAULT   = 0xFFB71C1C; // dark red

    // Receive stop-vibration broadcast so dismiss also stops vibration
    private final BroadcastReceiver stopVibrationReceiver = new BroadcastReceiver() {
        @Override
        public void onReceive(Context context, Intent intent) {
            // Already dismissed — nothing to do
        }
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        // Show over lock screen
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O_MR1) {
            setShowWhenLocked(true);
            setTurnScreenOn(true);
            KeyguardManager km = getSystemService(KeyguardManager.class);
            if (km != null) km.requestDismissKeyguard(this, null);
        } else {
            getWindow().addFlags(
                    WindowManager.LayoutParams.FLAG_SHOW_WHEN_LOCKED |
                            WindowManager.LayoutParams.FLAG_TURN_SCREEN_ON   |
                            WindowManager.LayoutParams.FLAG_DISMISS_KEYGUARD
            );
        }

        // Keep screen on while alert is showing
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        setContentView(R.layout.activity_alert_popup);

        String className  = getIntent().getStringExtra("className");
        float  confidence = getIntent().getFloatExtra("confidence", 0f);

        if (className == null) className = "Emergency";

        // Set background color based on class
        View rootView = findViewById(R.id.alertRootLayout);
        if (rootView != null) rootView.setBackgroundColor(colorForClass(className));

        // Set emoji
        TextView emojiView = findViewById(R.id.alertEmoji);
        if (emojiView != null) emojiView.setText(emojiForClass(className));

        // Set class name
        TextView classView = findViewById(R.id.alertClassName);
        if (classView != null) classView.setText(className.toUpperCase());

        // Set confidence
        TextView confView = findViewById(R.id.alertConfidence);
        if (confView != null) {
            if (className.equals("Loud Noise")) {
                confView.setText("Dangerous sound level detected");
            } else {
                confView.setText("Confidence: " + (int)(confidence * 100) + "%");
            }
        }

        // Dismiss button
        Button dismissBtn = findViewById(R.id.dismissButton);
        if (dismissBtn != null) {
            dismissBtn.setOnClickListener(v -> dismissAlert());
        }

        // Auto dismiss after 15 seconds
        handler = new Handler(Looper.getMainLooper());
        autoDismissRunnable = this::dismissAlert;
        handler.postDelayed(autoDismissRunnable, AUTO_DISMISS_MS);

        LocalBroadcastManager.getInstance(this)
                .registerReceiver(stopVibrationReceiver,
                        new IntentFilter(DetectionService.ACTION_STOP_VIBRATION));
    }

    private void dismissAlert() {
        // Stop vibration
        LocalBroadcastManager.getInstance(this)
                .sendBroadcast(new Intent(DetectionService.ACTION_STOP_VIBRATION));
        finish();
    }

    @Override
    protected void onNewIntent(Intent intent) {
        super.onNewIntent(intent);
        setIntent(intent);

        // Update UI for new alert without relaunching activity
        String className  = intent.getStringExtra("className");
        float  confidence = intent.getFloatExtra("confidence", 0f);
        if (className == null) className = "Emergency";

        View rootView = findViewById(R.id.alertRootLayout);
        if (rootView != null) rootView.setBackgroundColor(colorForClass(className));

        TextView emojiView = findViewById(R.id.alertEmoji);
        if (emojiView != null) emojiView.setText(emojiForClass(className));

        TextView classView = findViewById(R.id.alertClassName);
        if (classView != null) classView.setText(className.toUpperCase());

        TextView confView = findViewById(R.id.alertConfidence);
        if (confView != null) {
            if (className.equals("Loud Noise")) {
                confView.setText("Dangerous sound level detected");
            } else {
                confView.setText("Confidence: " + (int)(confidence * 100) + "%");
            }
        }

        // Reset auto dismiss timer
        if (handler != null && autoDismissRunnable != null) {
            handler.removeCallbacks(autoDismissRunnable);
            handler.postDelayed(autoDismissRunnable, AUTO_DISMISS_MS);
        }
    }

    @Override
    protected void onDestroy() {
        if (handler != null && autoDismissRunnable != null) {
            handler.removeCallbacks(autoDismissRunnable);
        }
        LocalBroadcastManager.getInstance(this)
                .unregisterReceiver(stopVibrationReceiver);
        super.onDestroy();
    }

    private int colorForClass(String className) {
        switch (className) {
            case "Glass Breaking": return COLOR_GLASS;
            case "Siren":          return COLOR_SIREN;
            case "Baby Crying":    return COLOR_BABY;
            case "Car Horn":       return COLOR_HORN;
            case "Explosion":      return COLOR_EXPLOSION;
            case "Loud Noise":     return COLOR_LOUD;
            default:               return COLOR_DEFAULT;
        }
    }

    private String emojiForClass(String className) {
        switch (className) {
            case "Glass Breaking": return "💥";
            case "Siren":          return "🚨";
            case "Baby Crying":    return "👶";
            case "Car Horn":       return "📯";
            case "Explosion":      return "🔥";
            case "Loud Noise":     return "🔊";
            default:               return "⚠️";
        }
    }
}