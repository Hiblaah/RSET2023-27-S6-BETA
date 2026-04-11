package com.emergency.classification;

import android.Manifest;
import android.content.Context;
import android.content.pm.PackageManager;
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaRecorder;
import android.util.Log;
import androidx.core.app.ActivityCompat;

public class AudioRecorder {
    private static final String TAG = "AudioRecorder";

    // YAMNet requires 16000 Hz — different from old app's 22050 Hz
    private static final int SAMPLE_RATE  = 16000;

    // YAMNet input size: 15600 samples = 0.975 seconds at 16000 Hz
    private static final int YAMNET_INPUT = AudioClassifier.YAMNET_INPUT_SIZE; // 15600

    // Sliding window: keep 3 seconds of history (48000 samples)
    // step forward by 1 second (16000 samples) each call.
    private static final int HISTORY_SIZE = SAMPLE_RATE * 3;  // 48000 samples
    private static final int STEP_SIZE    = SAMPLE_RATE;      // 16000 samples

    private AudioRecord audioRecord;
    private boolean     isRecording = false;
    private final Context context;

    // Rolling history buffer
    private final short[] history = new short[HISTORY_SIZE];
    private int stepCount = 0;

    public AudioRecorder(Context context) {
        this.context = context;
    }

    public boolean startRecording() {
        if (ActivityCompat.checkSelfPermission(context, Manifest.permission.RECORD_AUDIO)
                != PackageManager.PERMISSION_GRANTED) {
            Log.e(TAG, "No RECORD_AUDIO permission");
            return false;
        }

        int minBuf = AudioRecord.getMinBufferSize(
                SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT);
        int bufSize = Math.max(minBuf, STEP_SIZE * 2);

        try {
            audioRecord = new AudioRecord(
                    MediaRecorder.AudioSource.MIC,
                    SAMPLE_RATE,
                    AudioFormat.CHANNEL_IN_MONO,
                    AudioFormat.ENCODING_PCM_16BIT,
                    bufSize);

            if (audioRecord.getState() != AudioRecord.STATE_INITIALIZED) {
                Log.e(TAG, "AudioRecord failed to initialize");
                audioRecord.release();
                audioRecord = null;
                return false;
            }

            audioRecord.startRecording();
            isRecording = true;
            stepCount   = 0;
            Log.d(TAG, "Recording started at " + SAMPLE_RATE + " Hz");
            return true;

        } catch (SecurityException e) {
            Log.e(TAG, "SecurityException starting recorder", e);
            return false;
        }
    }

    /**
     * Blocking — reads exactly 1 second of fresh audio (16000 samples).
     * Returns float[] of YAMNET_INPUT_SIZE (15600) ready for YAMNet,
     * or null if history buffer is still warming up (first 3 seconds).
     * Audio converted from short PCM to float in [-1, 1] range.
     */
    public float[] recordAudio() {
        if (audioRecord == null || !isRecording) return null;

        short[] step = new short[STEP_SIZE];
        int totalRead = 0;
        while (totalRead < STEP_SIZE) {
            int read = audioRecord.read(step, totalRead, STEP_SIZE - totalRead);
            if (read < 0) {
                Log.w(TAG, "AudioRecord.read returned " + read);
                break;
            }
            totalRead += read;
        }

        // Slide history left, append new step on the right
        System.arraycopy(history, STEP_SIZE, history, 0, HISTORY_SIZE - STEP_SIZE);
        for (int i = 0; i < STEP_SIZE; i++) {
            history[HISTORY_SIZE - STEP_SIZE + i] = step[i];
        }

        stepCount++;

        // Wait until buffer is full (3 seconds) before returning windows
        if (stepCount < 3) {
            Log.d(TAG, "Warming up — step " + stepCount + "/3");
            return null;
        }

        // Convert most recent YAMNET_INPUT samples from short to float [-1, 1]
        float[] floatAudio = new float[YAMNET_INPUT];
        int offset = HISTORY_SIZE - YAMNET_INPUT;
        for (int i = 0; i < YAMNET_INPUT; i++) {
            floatAudio[i] = history[offset + i] / 32768.0f;
        }

        Log.d(TAG, "Sliding window ready: " + YAMNET_INPUT + " samples");
        return floatAudio;
    }

    public void release() {
        isRecording = false;
        stepCount   = 0;
        if (audioRecord != null) {
            try {
                if (audioRecord.getRecordingState() == AudioRecord.RECORDSTATE_RECORDING) {
                    audioRecord.stop();
                }
            } catch (IllegalStateException ignored) {}
            audioRecord.release();
            audioRecord = null;
        }
        Log.d(TAG, "AudioRecorder released");
    }

    public boolean isRecording() { return isRecording; }
    public int getSampleRate()   { return SAMPLE_RATE; }
}