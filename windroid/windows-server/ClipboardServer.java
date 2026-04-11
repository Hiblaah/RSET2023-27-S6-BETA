import java.io.*;
import java.net.*;
import java.awt.Toolkit;
import java.awt.datatransfer.StringSelection;
import javax.swing.SwingUtilities;
import java.awt.datatransfer.Transferable;
import java.awt.datatransfer.DataFlavor;
import java.awt.event.KeyEvent;

public class ClipboardServer {

    private static volatile boolean clipboardEnabled = true;
    private static volatile PrintWriter androidWriter = null;
    private static int lastBattery = -1;
    private static String lastMode = "";
    private static boolean lastCharging = false;
    private static boolean lastBt = false;
    private static volatile String lastClipboardText = "";

    // Track the last text we received FROM the phone so the monitor never
    // echoes it back. This replaces the fragile one-shot ignoreNextClipboard flag.
    private static volatile String lastReceivedFromPhone = null;

    public static void setClipboardEnabled(boolean val) {
        clipboardEnabled = val;
    }

    private static final Object writeLock = new Object();

    public static void sendToAndroid(String message) {
        synchronized (writeLock) {
            if (androidWriter != null) {
                androidWriter.println(message);
                androidWriter.flush();
                System.out.println("Sent to Android: " + message);
            } else {
                System.err.println("No Android connection");
            }
        }
    }

    public static void start(int port) {
        startClipboardMonitor();
        new Thread(() -> {
            try (ServerSocket serverSocket = new ServerSocket(port)) {
                serverSocket.setReuseAddress(true);
                System.out.println("Server listening on port " + port);
                while (true) {
                    // REPLACE WITH THIS:
                    Socket client = serverSocket.accept();

                    // WITH THIS:
                    if (androidWriter != null) {
                        if (WindowsServer.phoneConnected) {
                            System.out.println("Already connected — rejecting duplicate");
                            client.close();
                            continue;
                        } else {
                            // phoneConnected is false — the handleClient finally block already
                            // cleaned up, just null the writer to be safe
                            System.out.println("Stale writer found — accepting new connection");
                            androidWriter = null;
                        }
                    }

                    new Thread(() -> handleClient(client)).start();
                }
            } catch (BindException e) {
                System.err.println("Port already in use!");
            } catch (Exception e) {
                e.printStackTrace();
            }
        }).start();
    }

    public static void startClipboardMonitor() {
        new Thread(() -> {
            String lastSentToPhone = "";
            while (true) {
                try {
                    Thread.sleep(500);

                    Transferable contents = Toolkit.getDefaultToolkit()
                            .getSystemClipboard().getContents(null);

                    if (contents != null && contents.isDataFlavorSupported(DataFlavor.stringFlavor)) {
                        String text = (String) contents.getTransferData(DataFlavor.stringFlavor);
                        if (lastReceivedFromPhone != null && text.equals(lastReceivedFromPhone)) {
                            lastReceivedFromPhone = null;
                            lastClipboardText = text;
                            continue;
                        }

                        if (text.equals(lastClipboardText))
                            continue;

                        lastClipboardText = text;

                        System.out.println("Clipboard changed, sending to Android: "
                                + text.substring(0, Math.min(30, text.length())));

                        sendToAndroid("CLIPBOARD=" + text);
                    }
                } catch (Exception ignored) {
                }
            }
        }).start();
    }

    private static void handleClient(Socket client) {
        String incomingIp = client.getInetAddress().getHostAddress();
        String pairedFingerprint = WindowsServer.getPairedFingerprint();
        String phoneName;
        String fingerprint;
        BufferedReader in;

        try {
            in = new BufferedReader(new InputStreamReader(client.getInputStream()));
            String hello = in.readLine();
            if (hello == null || !hello.startsWith("HELLO|")) {
                System.out.println("Unknown client — no handshake");
                try {
                    client.close();
                } catch (Exception ignored) {
                }
                return;
            }
            String[] helloParts = hello.split("\\|", 3);
            phoneName = helloParts.length > 1 ? helloParts[1] : "Unknown";
            fingerprint = helloParts.length > 2 ? helloParts[2] : "";
            if (pairedFingerprint != null && !pairedFingerprint.equals(fingerprint)) {
                System.out.println("Rejected unknown phone: " + phoneName);
                try {
                    client.close();
                } catch (Exception ignored) {
                }
                return;
            }
        } catch (Exception e) {
            System.err.println("Handshake failed: " + e.getMessage());
            try {
                client.close();
            } catch (Exception ignored) {
            }
            return;
        }

        try {
            WindowsServer.onPhoneConnected(incomingIp, phoneName, fingerprint);
            androidWriter = new PrintWriter(client.getOutputStream(), true);
            String pcName = java.net.InetAddress.getLocalHost().getHostName();
            androidWriter.println("PC_NAME=" + pcName);
            MediaBridgeManager.start();
            System.out.println("Android persistent connection established from: " + incomingIp);

            String line;
            while ((line = in.readLine()) != null) {
                System.out.println("[DEBUG] PC received raw line: " + line);
                if (line.equals("PING")) {
                    androidWriter.println("PONG");
                    continue;
                }
                System.out.println("Received line: " + line);

                if (line.startsWith("CLIPBOARD=")) {
                    if (clipboardEnabled) {
                        String text = line.substring("CLIPBOARD=".length());
                        lastReceivedFromPhone = text;
                        lastClipboardText = text;
                        StringSelection selection = new StringSelection(text);
                        Toolkit.getDefaultToolkit().getSystemClipboard().setContents(selection, null);
                        System.out.println("Clipboard set on PC: " + text);
                    }
                } else if (line.startsWith("NOTIF|")) {
                    NotificationServer.handle(line);
                } else if (line.equals("MIRROR_STOPPED")) {
                    WindowsServer.onMirrorStopped();
                } else if (line.startsWith("STATUS|")) {
                    String[] parts = line.split("\\|");
                    int battery = -1;
                    String mode = "UNKNOWN";
                    boolean charging = false;
                    boolean bt = false;
                    for (String part : parts) {
                        if (part.startsWith("BATTERY:"))
                            battery = Integer.parseInt(part.substring(8));
                        if (part.startsWith("MODE:"))
                            mode = part.substring(5);
                        if (part.startsWith("CHARGING:"))
                            charging = Boolean.parseBoolean(part.substring(9));
                        if (part.startsWith("BT:"))
                            bt = Boolean.parseBoolean(part.substring(3));
                    }
                    if (battery == lastBattery && mode.equals(lastMode) && charging == lastCharging && bt == lastBt)
                        continue;
                    lastBattery = battery;
                    lastMode = mode;
                    lastCharging = charging;
                    lastBt = bt;
                    final int b = battery;
                    final String m = mode;
                    final boolean c = charging;
                    final boolean bluetooth = bt;
                    SwingUtilities.invokeLater(() -> {
                        if (WindowsServer.batteryLabel != null)
                            WindowsServer.batteryLabel
                                    .setText("Battery: " + b + "%" + (c ? " Charging" : " Not Charging"));
                        if (WindowsServer.modeLabel != null)
                            WindowsServer.modeLabel.setText("Mode: " + m + (bluetooth ? "  |  BT ON" : "  |  BT OFF"));
                    });
                } else if (line.startsWith("MOUSE_MOVE:")) {
                    String[] parts = line.substring("MOUSE_MOVE:".length()).split(",");
                    int dx = Integer.parseInt(parts[0]);
                    int dy = Integer.parseInt(parts[1]);
                    MouseController.move(dx, dy);
                } else if (line.equals("MOUSE_CLICK")) {
                    MouseController.click();
                } else if (line.equals("MOUSE_DOWN")) {
                    MouseController.mouseDown();
                } else if (line.equals("MOUSE_UP")) {
                    MouseController.mouseUp();
                } else if (line.equals("MOUSE_RIGHT_CLICK")) {
                    MouseController.rightClick();
                } else if (line.startsWith("MOUSE_SCROLL:")) {
                    int amount = Integer.parseInt(line.substring("MOUSE_SCROLL:".length()));
                    MouseController.scroll(amount);
                } else if (line.equals("CMD:LOCK_PC")) {
                    try {
                        Runtime.getRuntime().exec(new String[] { "rundll32.exe", "user32.dll,LockWorkStation" });
                        System.out.println("PC locked by phone");
                    } catch (Exception e) {
                        System.err.println("Failed to lock PC: " + e.getMessage());
                    }
                } else if (line.equals("CMD:SIGNOUT_PC")) {
                    try {
                        Runtime.getRuntime().exec(new String[] { "cmd.exe", "/c", "shutdown", "/l" });
                        System.out.println("PC sign-out triggered by phone");
                    } catch (Exception e) {
                        System.err.println("Failed to sign out PC: " + e.getMessage());
                    }
                } else if (line.equals("CMD:MEDIA_PLAY")) {
                    MediaBridgeManager.sendCommand("PLAY");
                } else if (line.equals("CMD:MEDIA_PAUSE")) {
                    MediaBridgeManager.sendCommand("PAUSE");
                } else if (line.equals("CMD:MEDIA_NEXT")) {
                    MediaBridgeManager.sendCommand("NEXT");
                } else if (line.equals("CMD:MEDIA_PREV")) {
                    MediaBridgeManager.sendCommand("PREV");
                } else if (line.startsWith("KEY_DOWN:")) {
                    String key = line.substring("KEY_DOWN:".length());
                    int code = KeyboardController.toKeyCode(key);
                    if (code != -1)
                        KeyboardController.keyDown(code);
                } else if (line.startsWith("KEY_UP:")) {
                    String key = line.substring("KEY_UP:".length());
                    int code = KeyboardController.toKeyCode(key);
                    if (code != -1)
                        KeyboardController.keyUp(code);
                } else if (line.startsWith("GYRO:")) {
                    String[] parts = line.substring("GYRO:".length()).split(",");
                    float dx = Float.parseFloat(parts[0]);
                    float dy = Float.parseFloat(parts[1]);
                    if (dy < -2.0f)
                        KeyboardController.keyDown(KeyEvent.VK_W);
                    else
                        KeyboardController.keyUp(KeyEvent.VK_W);
                    if (dy > 2.0f)
                        KeyboardController.keyDown(KeyEvent.VK_S);
                    else
                        KeyboardController.keyUp(KeyEvent.VK_S);
                    if (dx < -2.0f)
                        KeyboardController.keyDown(KeyEvent.VK_A);
                    else
                        KeyboardController.keyUp(KeyEvent.VK_A);
                    if (dx > 2.0f)
                        KeyboardController.keyDown(KeyEvent.VK_D);
                    else
                        KeyboardController.keyUp(KeyEvent.VK_D);
                } else if (line.startsWith("STEER:")) {
                    float steer = Float.parseFloat(line.substring("STEER:".length()));
                    KeyboardController.steer(steer);
                }

                if (line.startsWith("KEY:")) {
                    RemoteKeyboard.handle(line);
                }

                if (line.startsWith("FILE_REQ_LIST|") || line.startsWith("FILE_REQ_DOWNLOAD|")
                        || line.equals("FILE_REQ_DRIVES")) {
                    FileAccessHandler.handleMessage(line);
                    continue;
                }
                if (line.startsWith("ANDROID_RES_LIST|")) {
                    AndroidFileBrowser.updateList(line);
                    continue;
                }
                if (line.startsWith("ANDROID_RES_FILE|")) {
                    try {
                        String[] parts = line.split("\\|", 3);
                        byte[] data = java.util.Base64.getDecoder().decode(parts[2]);
                        java.nio.file.Files.write(
                                java.nio.file.Paths.get(AndroidFileBrowser.getDownloadPath(), parts[1]), data);
                        System.out.println("Downloaded from Android: " + parts[1]);
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                    continue;
                }
            }
        } catch (Exception e) {
            System.err.println("Connection error: " + e.getMessage());
        } finally {
            if (!client.isClosed()) {
                try {
                    client.close();
                } catch (Exception ignored) {
                }
            }
            androidWriter = null;
            MediaBridgeManager.stop();
            WindowsServer.onPhoneDisconnected();
        }
    }

}