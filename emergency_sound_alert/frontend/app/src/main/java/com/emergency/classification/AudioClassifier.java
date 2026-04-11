package com.emergency.classification;

import android.content.Context;
import android.util.Log;
import org.tensorflow.lite.Interpreter;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import android.content.res.AssetFileDescriptor;

public class AudioClassifier {
    private static final String TAG = "AudioClassifier";

    // YAMNet expects exactly 15600 samples (0.975 seconds at 16000 Hz)
    public static final int YAMNET_INPUT_SIZE = 15600;
    public static final int YAMNET_NUM_CLASSES = 521;

    // YAMNet class indices mapped to your 6 emergency classes
    // These are the AudioSet class indices YAMNet was trained on
    private static final int[] GLASS_BREAKING_INDICES = {435, 437};
    private static final int[] SIREN_INDICES          = {388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401};
    private static final int[] BABY_CRYING_INDICES    = {14, 15, 16, 17, 18, 19, 20, 21, 22};
    private static final int[] CAR_HORN_INDICES = {302, 303};
    private static final int[] EXPLOSION_INDICES = {358, 368, 412, 413, 418, 419, 431};
    private Interpreter interpreter;

    public static class PredictionResult {
        public final String className;
        public final float  confidence;
        public final int    classIndex; // 0=NonEmergency,1=Glass,2=Siren,3=Baby,4=CarHorn,5=Explosion
        public final float[] allProbabilities; // your 6 class scores

        public PredictionResult(String className, float confidence,
                                int classIndex, float[] allProbabilities) {
            this.className        = className;
            this.confidence       = confidence;
            this.classIndex       = classIndex;
            this.allProbabilities = allProbabilities;
        }

        public boolean isEmergency() {
            return classIndex > 0;
        }
    }

    public AudioClassifier(Context context) throws IOException {
        Interpreter.Options options = new Interpreter.Options();
        options.setNumThreads(2);
        interpreter = new Interpreter(loadModelFile(context, "yamnet.tflite"), options);
        Log.d(TAG, "YAMNet model loaded successfully");
    }

    private MappedByteBuffer loadModelFile(Context context, String modelName) throws IOException {
        AssetFileDescriptor afd = context.getAssets().openFd(modelName);
        FileInputStream fis = new FileInputStream(afd.getFileDescriptor());
        FileChannel fc = fis.getChannel();
        return fc.map(FileChannel.MapMode.READ_ONLY, afd.getStartOffset(), afd.getDeclaredLength());
    }

    /**
     * Run YAMNet inference on a 15600-sample float array (1 second at 16000 Hz).
     * YAMNet internally divides this into frames, runs its CNN, and averages
     * the per-frame scores into a single 521-class output.
     */
    public PredictionResult predict(float[] audioData) {
        float[][] input = new float[1][YAMNET_INPUT_SIZE];
        int copyLen = Math.min(audioData.length, YAMNET_INPUT_SIZE);
        System.arraycopy(audioData, 0, input[0], 0, copyLen);

        float[][] output = new float[1][YAMNET_NUM_CLASSES];

        interpreter.run(input, output);

        // ── DEBUG: log top 10 raw YAMNet indices ──────────────────────────────
        // Remove this block once correct indices are confirmed
        float[] scoresCopy = output[0].clone();
        StringBuilder sb = new StringBuilder("TOP SCORES: ");
        for (int n = 0; n < 10; n++) {
            float maxVal = 0f; int maxIdx = 0;
            for (int i = 0; i < scoresCopy.length; i++) {
                if (scoresCopy[i] > maxVal) { maxVal = scoresCopy[i]; maxIdx = i; }
            }
            sb.append(maxIdx).append("=").append(String.format("%.3f", maxVal)).append(" ");
            scoresCopy[maxIdx] = 0f;
        }
        Log.d("YAMNetDebug", sb.toString());
        // ── END DEBUG ─────────────────────────────────────────────────────────

        return mapToEmergencyClass(output[0]);
    }

    /**
     * Collapse YAMNet's 521 class scores into your 6 emergency classes.
     * For each of your classes, take the MAX score among all mapped YAMNet indices.
     * The class with the highest score wins.
     */
    private PredictionResult mapToEmergencyClass(float[] scores) {
        float[] emergencyScores = new float[6];

        // Class 0: Non-Emergency — will be set as remainder
        // Class 1: Glass Breaking
        emergencyScores[1] = maxScore(scores, GLASS_BREAKING_INDICES);
        // Class 2: Siren
        emergencyScores[2] = maxScore(scores, SIREN_INDICES);
        // Class 3: Baby Crying
        emergencyScores[3] = maxScore(scores, BABY_CRYING_INDICES);
        // Class 4: Car Horn
        emergencyScores[4] = maxScore(scores, CAR_HORN_INDICES);
        // Class 5: Explosion
        emergencyScores[5] = maxScore(scores, EXPLOSION_INDICES);

        // Non-emergency score = 1 minus the highest emergency score
        float maxEmergency = 0f;
        for (int i = 1; i < 6; i++) {
            if (emergencyScores[i] > maxEmergency) maxEmergency = emergencyScores[i];
        }
        emergencyScores[0] = Math.max(0f, 1.0f - maxEmergency);

        // Find winning class
        int    bestClass = 0;
        float  bestScore = emergencyScores[0];
        for (int i = 1; i < 6; i++) {
            if (emergencyScores[i] > bestScore) {
                bestScore = emergencyScores[i];
                bestClass = i;
            }
        }

        String className = getClassName(bestClass);
        Log.d(TAG, "YAMNet mapped → " + className
                + " (" + String.format("%.1f", bestScore * 100) + "%)"
                + " Glass=" + String.format("%.3f", emergencyScores[1])
                + " Siren=" + String.format("%.3f", emergencyScores[2])
                + " Baby="  + String.format("%.3f", emergencyScores[3])
                + " Horn="  + String.format("%.3f", emergencyScores[4])
                + " Expl="  + String.format("%.3f", emergencyScores[5]));

        return new PredictionResult(className, bestScore, bestClass, emergencyScores);
    }

    private float maxScore(float[] scores, int[] indices) {
        float max = 0f;
        for (int idx : indices) {
            if (idx < scores.length && scores[idx] > max) {
                max = scores[idx];
            }
        }
        return max;
    }

    private String getClassName(int index) {
        switch (index) {
            case 1: return "Glass Breaking";
            case 2: return "Siren";
            case 3: return "Baby Crying";
            case 4: return "Car Horn";
            case 5: return "Explosion";
            default: return "Non-Emergency";
        }
    }

    public void close() {
        if (interpreter != null) {
            interpreter.close();
            interpreter = null;
        }
    }
}