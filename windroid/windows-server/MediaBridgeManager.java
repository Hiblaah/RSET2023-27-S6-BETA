import java.io.*;

public class MediaBridgeManager {

    private static Process process = null;
    private static PrintWriter writer = null;

    public static void start() {
        new Thread(() -> {
            try {
                String currentDir = new File("").getAbsolutePath();
                File exeFile = new File(currentDir, "MediaBridge.exe");

                // If not found, try parent directory
                if (!exeFile.exists()) {
                    exeFile = new File(new File(currentDir).getParent(), "MediaBridge.exe");
                }

                ProcessBuilder pb = new ProcessBuilder(exeFile.getAbsolutePath());
                pb.directory(exeFile.getParentFile());
                pb.directory(new File(currentDir));
                process = pb.start();
                writer = new PrintWriter(process.getOutputStream(), true);

                System.out.println("MediaBridge started");

                BufferedReader reader = new BufferedReader(
                        new InputStreamReader(process.getInputStream()));

                String line;
                while ((line = reader.readLine()) != null) {
                    System.out.println("Media update: " + line);
                    ClipboardServer.sendToAndroid("MEDIA=" + line);
                }

            } catch (Exception e) {
                System.err.println("MediaBridge error: " + e.getMessage());
            }
        }).start();
    }

    public static void sendCommand(String command) {
        if (writer != null) {
            writer.println(command);
        } else {
            System.err.println("MediaBridge not running");
        }
    }

    public static void stop() {
        if (process != null) {
            process.destroy();
            process = null;
            writer = null;
        }
    }
}