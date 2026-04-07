import javax.swing.*;
import java.awt.*;
import java.awt.event.*;
import java.awt.image.BufferedImage;
import java.io.*;
import java.net.*;
import javax.imageio.ImageIO;
import java.util.concurrent.CountDownLatch;

public class ScreenMirrorServer {

    private static JFrame mirrorFrame;
    private static JLabel imageLabel;
    private static JLabel statusLabel;
    private static volatile boolean mirrorWindowCreated = false;
    private static int frameCount = 0;
    private static long lastFrameTime = 0;
    private static BufferedImage currentImage = null;
    private static volatile boolean enabled = false;

    // Signals that the ServerSocket is bound and ready before we tell Android to connect
    private static final CountDownLatch serverReady = new CountDownLatch(1);

    public static void setEnabled(boolean val) {
        enabled = val;
        if (!enabled && mirrorFrame != null) {
            SwingUtilities.invokeLater(() -> {
                mirrorFrame.setVisible(false);
                mirrorWindowCreated = false;
            });
        }
    }

    public static void start(int port) {
        new Thread(() -> {
            try (ServerSocket serverSocket = new ServerSocket(port)) {
                serverSocket.setSoTimeout(0);
                serverSocket.setReuseAddress(true);
                System.out.println("🖥️  Mirror server listening on port " + port);

                // Signal that we are bound and ready BEFORE accepting any connection
                serverReady.countDown();

                while (true) {
                    try {
                        Socket client = serverSocket.accept();
                        if (enabled) {
                            handleMirrorClient(client);
                        } else {
                            client.close();
                        }
                    } catch (Exception e) {
                        if (!(e instanceof SocketTimeoutException)) {
                            System.out.println("Mirror error: " + e.getMessage());
                        }
                    }
                }
            } catch (BindException e) {
                System.err.println("❌ Mirror port already in use!");
                serverReady.countDown(); // Unblock so requestMirrorStart doesn't hang
            } catch (Exception e) {
                e.printStackTrace();
                serverReady.countDown();
            }
        }, "MirrorServerThread").start();
    }

    /**
     * Call this to tell Android to start mirroring.
     * Waits until the ServerSocket is bound so the phone never connects to a closed port.
     */
    public static void requestMirrorStart() {
        new Thread(() -> {
            try {
                serverReady.await(); // Block until socket is ready
                ClipboardServer.sendToAndroid("CMD:START_MIRROR");
                System.out.println("📡 Sent START_MIRROR to Android");
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }, "MirrorStartThread").start();
    }

    public static void requestMirrorStop() {
        ClipboardServer.sendToAndroid("CMD:STOP_MIRROR");
    }

    private static void handleMirrorClient(Socket client) {
        try {
            client.setSoTimeout(5000);
            DataInputStream dis = new DataInputStream(client.getInputStream());

            byte[] header = new byte[6];
            dis.readFully(header);
            String headerStr = new String(header);

            if (headerStr.equals("FRAME:")) {
                int imageSize = dis.readInt();
                if (imageSize <= 0 || imageSize > 10 * 1024 * 1024) {
                    // Sanity check: reject obviously bad frame sizes
                    System.out.println("⚠️  Rejected frame with suspicious size: " + imageSize);
                    return;
                }

                byte[] imageData = new byte[imageSize];
                dis.readFully(imageData);

                BufferedImage image = ImageIO.read(new ByteArrayInputStream(imageData));

                if (image != null) {
                    frameCount++;
                    long currentTime = System.currentTimeMillis();

                    if (frameCount % 30 == 0 && lastFrameTime > 0) {
                        double fps = 30000.0 / (currentTime - lastFrameTime);
                        System.out.printf("📺 Frame #%d (%.1f FPS)%n", frameCount, fps);
                        lastFrameTime = currentTime;
                    }
                    if (frameCount % 30 == 0) lastFrameTime = currentTime;

                    displayMirrorImage(image);
                }
            }
        } catch (SocketTimeoutException | EOFException e) {
            // Normal disconnection
        } catch (Exception e) {
            if (e.getMessage() != null && !e.getMessage().contains("Connection reset")) {
                System.out.println("Frame error: " + e.getMessage());
            }
        } finally {
            try { client.close(); } catch (Exception ignored) {}
        }
    }

    private static void displayMirrorImage(BufferedImage image) {
        currentImage = image;

        SwingUtilities.invokeLater(() -> {
            if (!mirrorWindowCreated) {
                createMirrorWindow(image.getWidth(), image.getHeight());
            }

            int labelWidth  = imageLabel.getWidth();
            int labelHeight = imageLabel.getHeight();

            if (labelWidth > 0 && labelHeight > 0) {
                imageLabel.setIcon(new ImageIcon(getScaledImage(image, labelWidth, labelHeight)));
            } else {
                imageLabel.setIcon(new ImageIcon(image));
            }

            statusLabel.setText("Frame #" + frameCount + " | " +
                    image.getWidth() + "x" + image.getHeight());
        });
    }

    private static Image getScaledImage(BufferedImage image, int maxWidth, int maxHeight) {
        double scale = Math.min((double) maxWidth  / image.getWidth(),
                                (double) maxHeight / image.getHeight());
        return image.getScaledInstance(
                (int) (image.getWidth()  * scale),
                (int) (image.getHeight() * scale),
                Image.SCALE_SMOOTH);
    }

    private static void createMirrorWindow(int w, int h) {
        mirrorFrame = new JFrame("Windroid - Screen Mirror");
        mirrorFrame.setLayout(new BorderLayout());

        imageLabel = new JLabel() {
            @Override
            protected void paintComponent(Graphics g) {
                super.paintComponent(g);
                if (currentImage != null && getWidth() > 0 && getHeight() > 0) {
                    Image scaled = getScaledImage(currentImage, getWidth(), getHeight());
                    int x = (getWidth()  - scaled.getWidth(null))  / 2;
                    int y = (getHeight() - scaled.getHeight(null)) / 2;
                    g.drawImage(scaled, x, y, null);
                }
            }
        };
        imageLabel.setHorizontalAlignment(JLabel.CENTER);
        imageLabel.setBackground(Color.BLACK);
        imageLabel.setOpaque(true);
        imageLabel.setPreferredSize(new Dimension(w, h));

        statusLabel = new JLabel("Receiving frames...");
        statusLabel.setHorizontalAlignment(JLabel.CENTER);
        statusLabel.setBorder(BorderFactory.createEmptyBorder(5, 5, 5, 5));
        statusLabel.setBackground(Color.DARK_GRAY);
        statusLabel.setForeground(Color.WHITE);
        statusLabel.setOpaque(true);

        mirrorFrame.add(imageLabel, BorderLayout.CENTER);
        mirrorFrame.add(statusLabel, BorderLayout.SOUTH);

        mirrorFrame.addComponentListener(new ComponentAdapter() {
            public void componentResized(ComponentEvent evt) { imageLabel.repaint(); }
        });

        mirrorFrame.setDefaultCloseOperation(JFrame.HIDE_ON_CLOSE);
        mirrorFrame.setSize(w, h + 50);
        mirrorFrame.setLocationRelativeTo(null);
        mirrorFrame.setVisible(true);
        mirrorWindowCreated = true;

        System.out.println("🖥️  Mirror window opened: " + w + "x" + h);
    }
}