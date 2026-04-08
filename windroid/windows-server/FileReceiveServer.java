import javax.swing.*;
import java.awt.*;
import java.io.*;
import java.net.*;
import java.nio.file.*;

public class FileReceiveServer {

    private static final int FILE_TRANSFER_PORT = 6790;
    private static final String TAG = "FileReceiveServer";

    public static void start() {
        new Thread(() -> {
            try {
                ServerSocket serverSocket = new ServerSocket(FILE_TRANSFER_PORT);
                System.out.println("FileReceiveServer listening on port " + FILE_TRANSFER_PORT);

                while (true) {
                    Socket client = serverSocket.accept();
                    new Thread(() -> receiveFile(client)).start();
                }
            } catch (Exception e) {
                System.err.println(TAG + ": " + e.getMessage());
            }
        }).start();
    }

    private static void receiveFile(Socket socket) {
        JDialog progressDialog = new JDialog((Frame) null, "Receiving File", false);
        JProgressBar progressBar = new JProgressBar(0, 100);
        JLabel progressLabel = new JLabel("Receiving...");

        SwingUtilities.invokeLater(() -> {
            progressDialog.setSize(360, 120);
            progressDialog.setLocationRelativeTo(null);
            progressDialog.setDefaultCloseOperation(JDialog.DO_NOTHING_ON_CLOSE);
            progressDialog.getContentPane().setBackground(Color.decode("#1E1E1E"));

            JPanel inner = new JPanel();
            inner.setLayout(new BoxLayout(inner, BoxLayout.Y_AXIS));
            inner.setBackground(Color.decode("#1E1E1E"));
            inner.setBorder(BorderFactory.createEmptyBorder(16, 20, 16, 20));

            progressLabel.setForeground(Color.WHITE);
            progressLabel.setFont(new Font("Segoe UI", Font.PLAIN, 12));
            progressLabel.setAlignmentX(Component.LEFT_ALIGNMENT);

            progressBar.setStringPainted(true);
            progressBar.setForeground(Color.decode("#4CAF50"));
            progressBar.setBackground(Color.decode("#2A2A2A"));
            progressBar.setAlignmentX(Component.LEFT_ALIGNMENT);
            progressBar.setMaximumSize(new Dimension(Integer.MAX_VALUE, 20));

            inner.add(progressLabel);
            inner.add(Box.createVerticalStrut(10));
            inner.add(progressBar);
            progressDialog.add(inner, BorderLayout.CENTER);
            progressDialog.setVisible(true);
        });

        try {
            DataInputStream dis = new DataInputStream(socket.getInputStream());

            int nameLen = dis.readInt();
            byte[] nameBytes = new byte[nameLen];
            dis.readFully(nameBytes);
            String fileName = new String(nameBytes, "UTF-8");
            long fileSize = dis.readLong();

            SwingUtilities.invokeLater(() -> progressLabel.setText("Receiving: " + fileName));

            Path downloadsDir = Paths.get(System.getProperty("user.home"), "Downloads");
            Files.createDirectories(downloadsDir);
            File outFile = downloadsDir.resolve(fileName).toFile();

            try (FileOutputStream fos = new FileOutputStream(outFile)) {
                byte[] buffer = new byte[65536];
                long received = 0;

                while (received < fileSize) {
                    int toRead = (int) Math.min(buffer.length, fileSize - received);
                    int read = dis.read(buffer, 0, toRead);
                    if (read == -1) break;
                    fos.write(buffer, 0, read);
                    received += read;

                    final long finalReceived = received;
                    final int pct = fileSize > 0 ? (int) ((received * 100) / fileSize) : 0;
                    SwingUtilities.invokeLater(() -> {
                        progressBar.setValue(pct);
                        progressBar.setString(pct + "%  (" + toReadable(finalReceived) + " / " + toReadable(fileSize) + ")");
                    });
                }
            }

            socket.close();
            System.out.println(TAG + ": Saved " + fileName + " to Downloads");

            SwingUtilities.invokeLater(() -> {
                progressDialog.dispose();
                JOptionPane.showMessageDialog(null,
                        fileName + " saved to Downloads.",
                        "File Received",
                        JOptionPane.INFORMATION_MESSAGE);
            });

        } catch (Exception e) {
            System.err.println(TAG + ": Receive error: " + e.getMessage());
            SwingUtilities.invokeLater(progressDialog::dispose);
        }
    }

    private static String toReadable(long bytes) {
        if (bytes < 1024) return bytes + " B";
        if (bytes < 1024 * 1024) return String.format("%.1f KB", bytes / 1024.0);
        return String.format("%.1f MB", bytes / (1024.0 * 1024));
    }
}