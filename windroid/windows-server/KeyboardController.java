import java.awt.Robot;
import java.awt.event.KeyEvent;

public class KeyboardController {

    private static Robot robot;

    static {
        try {
            robot = new Robot();
        } catch (Exception e) {
            System.err.println("Robot init failed: " + e.getMessage());
        }
    }

    public static void keyDown(int keyCode) {
        if (robot != null) robot.keyPress(keyCode);
    }

    public static void keyUp(int keyCode) {
        if (robot != null) robot.keyRelease(keyCode);
    }
    private static volatile float currentSteer = 0f;
private static volatile boolean steerRunning = false;

public static void steer(float value) {
    currentSteer = value;

    if (steerRunning) return;
    steerRunning = true;

    new Thread(() -> {
        while (true) {
            try {
                float s = currentSteer;

                if (Math.abs(s) < 0.5f) {
                    // Centered — release and stop thread
                    keyUp(KeyEvent.VK_A);
                    keyUp(KeyEvent.VK_D);
                    steerRunning = false;
                    break;
                }

                int key     = s < 0 ? KeyEvent.VK_A : KeyEvent.VK_D;
                int opposite = s < 0 ? KeyEvent.VK_D : KeyEvent.VK_A;

                // Normalize intensity 0.0–1.0
                float intensity = Math.min((Math.abs(s) - 0.5f) / 7f, 1.0f);

                // Fixed cycle of 100ms, hold portion scales with intensity
                long holdMs  = (long)(intensity * 100);
                long restMs  = 100 - holdMs;

                keyUp(opposite);
                keyDown(key);
                Thread.sleep(Math.max(holdMs, 10));
                keyUp(key);
                if (restMs > 10) Thread.sleep(restMs);

            } catch (InterruptedException e) {
                break;
            }
        }
    }).start();
}
    public static int toKeyCode(String key) {
        return switch (key) {
            case "W", "UP"    -> KeyEvent.VK_W;
            case "A", "LEFT"  -> KeyEvent.VK_A;
            case "S", "DOWN"  -> KeyEvent.VK_S;
            case "D", "RIGHT" -> KeyEvent.VK_D;
            case "SPACE"      -> KeyEvent.VK_SPACE;
            case "ENTER"      -> KeyEvent.VK_ENTER;
            case "SHIFT"      -> KeyEvent.VK_SHIFT;
            case "E"          -> KeyEvent.VK_E;
            case "Q"          -> KeyEvent.VK_Q;
            case "R"          -> KeyEvent.VK_R;
            case "F"          -> KeyEvent.VK_F;
            case "Z"          -> KeyEvent.VK_Z;
            case "X"          -> KeyEvent.VK_X;
            case "ESC"        -> KeyEvent.VK_ESCAPE;
            case "ARR_UP"     -> KeyEvent.VK_UP;
            case "ARR_DOWN"   -> KeyEvent.VK_DOWN;
            case "ARR_LEFT"   -> KeyEvent.VK_LEFT;
            case "ARR_RIGHT"  -> KeyEvent.VK_RIGHT;
            default           -> -1;
        };
    }
}