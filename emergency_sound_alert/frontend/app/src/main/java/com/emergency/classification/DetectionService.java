package com.emergency.classification;

import android.app.Notification;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.app.Service;
import android.content.Intent;
import android.content.IntentFilter;
import android.os.Build;
import androidx.localbroadcastmanager.content.LocalBroadcastManager;
import android.os.IBinder;
import android.util.Log;
import androidx.core.app.NotificationCompat;
import java.io.IOException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;

public class DetectionService extends Service {
    private static final String TAG = "DetectionService";

    public static final String ACTION_STOP_VIBRATION   = "com.emergency.classification.STOP_VIBRATION";
    public static final String ACTION_DETECTION_RESULT = "com.emergency.classification.DETECTION_RESULT";
    public static final String EXTRA_CLASS_NAME        = "className";
    public static final String EXTRA_CONFIDENCE        = "confidence";
    public static final String EXTRA_IS_EMERGENCY      = "isEmergency";
    public static final String EXTRA_IS_DB_ALERT       = "isDbAlert";

    private static final int MONITOR_NOTIF_ID = 1;
    private static final int ALERT_NOTIF_ID   = 2;
    private static final int DB_NOTIF_ID      = 3;

    // dB threshold — fires on loudness alone regardless of ML class
    private static final double DB_THRESHOLD = 76.0;

    // YAMNet confidence thresholds
    private static final float ML_CONFIDENCE_THRESHOLD      = 0.20f;
    private static final float BABY_SINGLE_WINDOW_THRESHOLD = 0.20f;
    private static final float EXPLOSION_THRESHOLD          = 0.10f;
    private static final float CAR_HORN_THRESHOLD           = 0.20f;

    // Cooldown constants
    private static final long ML_ALERT_COOLDOWN_MS = 5000;
    private static final long DB_ALERT_COOLDOWN_MS = 8000;

    // Silence gate
    private static final double SILENCE_RMS_THRESHOLD = 300.0;

    // Per-class voting
    private static final int VOTE_WINDOW    = 3;
    private static final int VOTE_THRESHOLD = 2;

    // dB sustained detection — prevents door slams / claps from firing
    private static final int DB_SUSTAINED_WINDOWS = 3;
    private static final int DB_QUIET_TOLERANCE   = 2;

    private final int[]   voteBuffer = new int[VOTE_WINDOW];
    private final float[] voteConf   = new float[VOTE_WINDOW];
    private int           voteIndex  = 0;
    private int           voteFilled = 0;

    private long   lastMlAlertTime = 0;
    private long   lastDbAlertTime = 0;
    private int    dbLoudCount     = 0;
    private int    dbQuietCount    = 0;
    private String lastAlertClass  = "";

    private AudioClassifier classifier;
    private AudioRecorder   audioRecorder;
    private AlertManager    alertManager;

    private ExecutorService     executor;
    private final AtomicBoolean isRunning = new AtomicBoolean(false);

    private final android.content.BroadcastReceiver stopVibrationReceiver =
            new android.content.BroadcastReceiver() {
                @Override
                public void onReceive(android.content.Context context, Intent intent) {
                    if (ACTION_STOP_VIBRATION.equals(intent.getAction())) {
                        if (alertManager != null) {
                            alertManager.stopVibration();
                            Log.d(TAG, "Vibration stopped via broadcast");
                        }
                    }
                }
            };

    @Override
    public void onCreate() {
        super.onCreate();
        Log.d(TAG, "=== SERVICE onCreate ===");
        try {
            classifier   = new AudioClassifier(this);
            alertManager = new AlertManager(this);
            Log.d(TAG, "All components initialized");
        } catch (IOException e) {
            Log.e(TAG, "FATAL: classifier init failed", e);
            stopSelf(); return;
        } catch (Exception e) {
            Log.e(TAG, "FATAL: init error", e);
            stopSelf(); return;
        }
        LocalBroadcastManager.getInstance(this)
                .registerReceiver(stopVibrationReceiver,
                        new IntentFilter(ACTION_STOP_VIBRATION));
    }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        startForeground(MONITOR_NOTIF_ID,
                buildMonitorNotification("Listening for emergency sounds..."));
        if (isRunning.compareAndSet(false, true)) {
            audioRecorder = new AudioRecorder(this);
            if (!audioRecorder.startRecording()) {
                updateMonitorNotification("Error: microphone unavailable");
                stopSelf();
                return START_NOT_STICKY;
            }
            executor = Executors.newSingleThreadExecutor();
            executor.execute(this::detectionLoop);
        }
        return START_STICKY;
    }

    private void detectionLoop() {
        Log.d(TAG, "detectionLoop: entered");
        while (isRunning.get()) {
            try {
                float[] audioData = audioRecorder.recordAudio();

                if (audioData == null) {
                    updateMonitorNotification("Warming up...");
                    continue;
                }

                double rms = calculateRms(audioData);
                double db  = rmsToDb(rms);

                // Silence gate
                if (rms < SILENCE_RMS_THRESHOLD) {
                    Log.d(TAG, "Silence gate: skipping model (RMS="
                            + String.format("%.1f", rms) + ")");
                    resetVoteBuffer();
                    dbLoudCount  = 0;
                    dbQuietCount = 0;
                    updateMonitorNotification("Listening for emergency sounds...");
                    continue;
                }

                AudioClassifier.PredictionResult result = classifier.predict(audioData);

                Log.d(TAG, "Prediction: " + result.className
                        + " (" + (int)(result.confidence * 100) + "%)"
                        + " RMS=" + String.format("%.1f", rms)
                        + " dB="  + String.format("%.1f", db));
                Log.d(TAG, "All probs — "
                        + "NonEmergency:" + String.format("%.3f", result.allProbabilities[0])
                        + " Glass:"       + String.format("%.3f", result.allProbabilities[1])
                        + " Siren:"       + String.format("%.3f", result.allProbabilities[2])
                        + " Baby:"        + String.format("%.3f", result.allProbabilities[3])
                        + " CarHorn:"     + String.format("%.3f", result.allProbabilities[4])
                        + " Explosion:"   + String.format("%.3f", result.allProbabilities[5]));

                long now = System.currentTimeMillis();

                boolean shouldAlert = false;
                String  alertClass  = result.className;
                float   alertConf   = result.confidence;

                // ── Baby Crying: single window lower threshold ────────────────
                if (result.allProbabilities[3] > BABY_SINGLE_WINDOW_THRESHOLD) {
                    shouldAlert = true;
                    alertClass  = "Baby Crying";
                    alertConf   = result.allProbabilities[3];
                    Log.d(TAG, "Baby Crying — single window fire (prob="
                            + String.format("%.3f", result.allProbabilities[3]) + ")");

                } else {
                    // ── Check each emergency class directly by raw probability ─
                    int   bestEmergencyClass = 0;
                    float bestEmergencyScore = 0f;

                    if (result.allProbabilities[1] > bestEmergencyScore) { bestEmergencyScore = result.allProbabilities[1]; bestEmergencyClass = 1; }
                    if (result.allProbabilities[2] > bestEmergencyScore) { bestEmergencyScore = result.allProbabilities[2]; bestEmergencyClass = 2; }
                    if (result.allProbabilities[4] > CAR_HORN_THRESHOLD  && result.allProbabilities[4] > bestEmergencyScore) { bestEmergencyScore = result.allProbabilities[4]; bestEmergencyClass = 4; }
                    if (result.allProbabilities[5] > EXPLOSION_THRESHOLD && result.allProbabilities[5] > bestEmergencyScore) { bestEmergencyScore = result.allProbabilities[5]; bestEmergencyClass = 5; }

                    if (bestEmergencyScore > ML_CONFIDENCE_THRESHOLD) {
                        if (isInstantaneous(bestEmergencyClass)) {
                            // ── Explosion / Glass Breaking: fire immediately ───
                            shouldAlert = true;
                            alertClass  = getClassNameForIndex(bestEmergencyClass);
                            alertConf   = bestEmergencyScore;
                            Log.d(TAG, "Instantaneous sound — firing immediately: " + alertClass);

                        } else {
                            // ── Siren / Car Horn: require 2/3 votes ───────────
                            voteBuffer[voteIndex] = bestEmergencyClass;
                            voteConf[voteIndex]   = bestEmergencyScore;
                            voteIndex             = (voteIndex + 1) % VOTE_WINDOW;
                            if (voteFilled < VOTE_WINDOW) voteFilled++;

                            if (voteFilled >= VOTE_WINDOW) {
                                int[]   counts  = new int[6];
                                float[] sumConf = new float[6];
                                for (int i = 0; i < VOTE_WINDOW; i++) {
                                    counts[voteBuffer[i]]++;
                                    sumConf[voteBuffer[i]] += voteConf[i];
                                }
                                for (int cls = 1; cls < 6; cls++) {
                                    if (counts[cls] >= VOTE_THRESHOLD) {
                                        shouldAlert = true;
                                        alertClass  = getClassNameForIndex(cls);
                                        alertConf   = sumConf[cls] / counts[cls];
                                        Log.d(TAG, "Sustained sound voted: " + alertClass
                                                + " (" + counts[cls] + "/" + VOTE_WINDOW + ")");
                                        break;
                                    }
                                }
                            }
                        }
                    } else {
                        // No emergency class above threshold — push non-emergency
                        // into vote buffer to displace stale emergency votes
                        voteBuffer[voteIndex] = 0;
                        voteConf[voteIndex]   = 0f;
                        voteIndex             = (voteIndex + 1) % VOTE_WINDOW;
                        if (voteFilled < VOTE_WINDOW) voteFilled++;
                    }
                }

                // ── Fire ML alert ─────────────────────────────────────────────
                if (shouldAlert) {
                    boolean cooldownExpired = (now - lastMlAlertTime >= ML_ALERT_COOLDOWN_MS);
                    boolean classChanged    = !alertClass.equals(lastAlertClass);

                    if (cooldownExpired || classChanged) {
                        lastMlAlertTime = now;
                        lastAlertClass  = alertClass;
                        Log.d(TAG, "EMERGENCY (ML): " + alertClass);
                        broadcastResult(alertClass, alertConf, true);
                        showEmergencyNotification(alertClass, alertConf);
                        alertManager.triggerAlert(alertClass, null);
                        updateMonitorNotification("🚨 " + alertClass + " detected!");
                        resetVoteBuffer();
                    } else {
                        Log.d(TAG, "ML alert suppressed — cooldown active");
                        broadcastResult(result.className, result.confidence, false);
                    }

                } else if (db > DB_THRESHOLD) {
                    // ── dB sustained check ────────────────────────────────────
                    // Requires 3 consecutive loud windows with up to 2 quiet
                    // windows tolerated between them. Prevents door slams and
                    // clapping from triggering while still catching dog barking,
                    // party music, and other sustained loud sounds.
                    dbLoudCount++;
                    dbQuietCount = 0;
                    if (dbLoudCount >= DB_SUSTAINED_WINDOWS) {
                        if (now - lastDbAlertTime >= DB_ALERT_COOLDOWN_MS) {
                            lastDbAlertTime = now;
                            dbLoudCount     = 0;
                            dbQuietCount    = 0;
                            Log.d(TAG, "DB ALERT: " + String.format("%.1f", db) + " dB");
                            broadcastDbAlert(db);
                            showDbPopup(db);
                            showDbNotification(db);
                            alertManager.triggerDbAlert();
                            updateMonitorNotification("⚠️ Loud noise: " + String.format("%.0f", db) + " dB");
                        } else {
                            updateMonitorNotification("Loud noise: " + String.format("%.0f", db) + " dB");
                        }
                    }

                } else {
                    // Below dB threshold — apply quiet tolerance before resetting
                    dbQuietCount++;
                    if (dbQuietCount > DB_QUIET_TOLERANCE) {
                        dbLoudCount  = 0;
                        dbQuietCount = 0;
                    }
                    broadcastResult(result.className, result.confidence, false);
                    updateMonitorNotification("Last: " + result.className
                            + " " + (int)(result.confidence * 100) + "%");
                }

            } catch (Exception e) {
                if (isRunning.get()) Log.e(TAG, "detectionLoop error", e);
            }
        }
        Log.d(TAG, "detectionLoop: exited cleanly");
    }

    private boolean isInstantaneous(int classIndex) {
        return classIndex == 1 || classIndex == 5;
    }

    private void resetVoteBuffer() {
        for (int i = 0; i < VOTE_WINDOW; i++) {
            voteBuffer[i] = 0;
            voteConf[i]   = 0f;
        }
        voteIndex  = 0;
        voteFilled = 0;
    }

    private String getClassNameForIndex(int index) {
        switch (index) {
            case 1: return "Glass Breaking";
            case 2: return "Siren";
            case 3: return "Baby Crying";
            case 4: return "Car Horn";
            case 5: return "Explosion";
            default: return "Non-Emergency";
        }
    }

    // RMS from float audio [-1,1] scaled to match SILENCE_RMS_THRESHOLD
    private double calculateRms(float[] audioData) {
        double sumSq = 0;
        for (float s : audioData) sumSq += s * s;
        return Math.sqrt(sumSq / audioData.length) * 32768.0;
    }

    private double rmsToDb(double rms) {
        if (rms < 1.0) rms = 1.0;
        return 20.0 * Math.log10(rms / 32768.0) + 90.0;
    }

    private void broadcastResult(String className, float confidence, boolean isActive) {
        Intent intent = new Intent(ACTION_DETECTION_RESULT);
        intent.putExtra(EXTRA_CLASS_NAME,   className);
        intent.putExtra(EXTRA_CONFIDENCE,   confidence);
        intent.putExtra(EXTRA_IS_EMERGENCY, isActive);
        intent.putExtra(EXTRA_IS_DB_ALERT,  false);
        LocalBroadcastManager.getInstance(this).sendBroadcast(intent);
    }

    private void broadcastDbAlert(double db) {
        Intent intent = new Intent(ACTION_DETECTION_RESULT);
        intent.putExtra(EXTRA_CLASS_NAME,   "Loud Noise");
        intent.putExtra(EXTRA_CONFIDENCE,   1.0f);
        intent.putExtra(EXTRA_IS_EMERGENCY, true);
        intent.putExtra(EXTRA_IS_DB_ALERT,  true);
        LocalBroadcastManager.getInstance(this).sendBroadcast(intent);
    }

    private void showDbPopup(double db) {
        Intent popupIntent = new Intent(this, AlertPopupActivity.class);
        popupIntent.putExtra("className",  "Loud Noise");
        popupIntent.putExtra("confidence", 1.0f);
        popupIntent.setFlags(Intent.FLAG_ACTIVITY_NEW_TASK |
                Intent.FLAG_ACTIVITY_CLEAR_TOP | Intent.FLAG_ACTIVITY_SINGLE_TOP);
        startActivity(popupIntent);
    }

    private void showEmergencyNotification(String className, float confidence) {
        Intent fullScreenIntent = new Intent(this, AlertPopupActivity.class);
        fullScreenIntent.putExtra("className",  className);
        fullScreenIntent.putExtra("confidence", confidence);
        fullScreenIntent.setFlags(Intent.FLAG_ACTIVITY_NEW_TASK |
                Intent.FLAG_ACTIVITY_CLEAR_TOP | Intent.FLAG_ACTIVITY_SINGLE_TOP);

        PendingIntent fullScreenPI = PendingIntent.getActivity(
                this, (int) System.currentTimeMillis(), fullScreenIntent,
                PendingIntent.FLAG_IMMUTABLE | PendingIntent.FLAG_UPDATE_CURRENT);

        Notification n = new NotificationCompat.Builder(this, EmergencyApp.ALERT_CHANNEL_ID)
                .setContentTitle("🚨 EMERGENCY: " + className)
                .setContentText(emojiFor(className) + " " + className
                        + " — " + (int)(confidence * 100) + "%")
                .setSmallIcon(android.R.drawable.ic_dialog_alert)
                .setPriority(NotificationCompat.PRIORITY_MAX)
                .setCategory(NotificationCompat.CATEGORY_ALARM)
                .setFullScreenIntent(fullScreenPI, true)
                .setAutoCancel(true)
                .build();

        getSystemService(NotificationManager.class).notify(ALERT_NOTIF_ID, n);

        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.UPSIDE_DOWN_CAKE) {
            startActivity(fullScreenIntent);
        } else {
            NotificationManager nm = getSystemService(NotificationManager.class);
            if (nm.canUseFullScreenIntent()) startActivity(fullScreenIntent);
        }
    }

    private void showDbNotification(double db) {
        PendingIntent pi = PendingIntent.getActivity(this, 0,
                new Intent(this, MainActivity.class)
                        .setFlags(Intent.FLAG_ACTIVITY_SINGLE_TOP),
                PendingIntent.FLAG_IMMUTABLE | PendingIntent.FLAG_UPDATE_CURRENT);

        getSystemService(NotificationManager.class).notify(DB_NOTIF_ID,
                new NotificationCompat.Builder(this, EmergencyApp.DB_CHANNEL_ID)
                        .setContentTitle("⚠️ Very Loud Sound Detected")
                        .setContentText("Sound level: " + String.format("%.0f", db) + " dB")
                        .setSmallIcon(android.R.drawable.ic_dialog_alert)
                        .setPriority(NotificationCompat.PRIORITY_HIGH)
                        .setCategory(NotificationCompat.CATEGORY_ALARM)
                        .setAutoCancel(true)
                        .setContentIntent(pi)
                        .setVibrate(new long[]{0, 300, 150, 300})
                        .build());
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
    public void onDestroy() {
        isRunning.set(false);
        LocalBroadcastManager.getInstance(this)
                .unregisterReceiver(stopVibrationReceiver);
        if (executor      != null) { executor.shutdownNow();  executor      = null; }
        if (audioRecorder != null) { audioRecorder.release(); audioRecorder = null; }
        if (classifier    != null) { classifier.close();      classifier    = null; }
        super.onDestroy();
    }

    @Override public IBinder onBind(Intent intent) { return null; }

    private Notification buildMonitorNotification(String text) {
        return new NotificationCompat.Builder(this, EmergencyApp.MONITOR_CHANNEL_ID)
                .setContentTitle("Emergency Classification")
                .setContentText(text)
                .setSmallIcon(android.R.drawable.ic_btn_speak_now)
                .setContentIntent(PendingIntent.getActivity(this, 0,
                        new Intent(this, MainActivity.class),
                        PendingIntent.FLAG_IMMUTABLE))
                .setOngoing(true)
                .build();
    }

    private void updateMonitorNotification(String text) {
        getSystemService(NotificationManager.class)
                .notify(MONITOR_NOTIF_ID, buildMonitorNotification(text));
    }
}