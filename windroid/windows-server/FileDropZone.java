import javax.swing.*;
import java.awt.*;
import java.awt.datatransfer.*;
import java.awt.dnd.*;
import java.io.*;
import java.net.Socket;
import java.util.ArrayList;
import java.util.List;

public class FileDropZone {

    private static final int FILE_TRANSFER_PORT = 6789;
    private static JDialog progressDialog;
    private static JProgressBar progressBar;
    private static JLabel progressLabel;

    public static void attach(JComponent component) {
        component.setDropTarget(new DropTarget(component, new DropTargetListener() {

            @Override
            public void dragEnter(DropTargetDragEvent e) {
                if (WindowsServer.phoneConnected && isFileDrop(e.getTransferable())) {
                    e.acceptDrag(DnDConstants.ACTION_COPY);
                    component.setBorder(BorderFactory.createLineBorder(Color.decode("#4CAF50"), 2));
                }
            }

            @Override
            public void dragOver(DropTargetDragEvent e) {}

            @Override
            public void dropActionChanged(DropTargetDragEvent e) {}

            @Override
            public void dragExit(DropTargetEvent e) {
                component.setBorder(null);
            }

            @Override
            public void drop(DropTargetDropEvent e) {
                component.setBorder(null);

                if (!WindowsServer.phoneConnected) {
                    JOptionPane.showMessageDialog(null,
                            "No phone connected.", "Windroid", JOptionPane.WARNING_MESSAGE);
                    e.rejectDrop();
                    return;
                }

                try {
                    e.acceptDrop(DnDConstants.ACTION_COPY);
                    Transferable t = e.getTransferable();
                    List<File> files = (List<File>) t.getTransferData(DataFlavor.javaFileListFlavor);

                    if (files == null || files.isEmpty()) {
                        e.dropComplete(false);
                        return;
                    }

                    e.dropComplete(true);

                    new Thread(() -> {
                        List<File> allFiles = new ArrayList<>();
                        for (File file : files) {
                            collectFiles(file, allFiles);
                        }

                        int total = allFiles.size();
                        for (int i = 0; i < total; i++) {
                            File file = allFiles.get(i);
                            final int index = i + 1;
                            try {
                                sendFile(file, file.getName(), index, total);
                            } catch (Exception ex) {
                                SwingUtilities.invokeLater(() ->
                                        JOptionPane.showMessageDialog(null,
                                                "Failed to send: " + file.getName() + "\n" + ex.getMessage(),
                                                "Windroid", JOptionPane.ERROR_MESSAGE));
                            }
                        }
                    }).start();

                } catch (Exception ex) {
                    ex.printStackTrace();
                    e.dropComplete(false);
                }
            }
        }));
    }

    private static boolean isFileDrop(Transferable t) {
        return t.isDataFlavorSupported(DataFlavor.javaFileListFlavor);
    }

    private static void collectFiles(File file, List<File> result) {
        if (file.isDirectory()) {
            File[] children = file.listFiles();
            if (children != null) {
                for (File child : children) {
                    collectFiles(child, result);
                }
            }
        } else {
            result.add(file);
        }
    }

    private static void sendFile(File file, String displayName, int index, int total) throws IOException {
        String androidIp = WindowsServer.androidIp;
        if (androidIp == null) throw new IOException("No phone IP");

        long fileSize = file.length();
        String label = total > 1 ? displayName + " (" + index + "/" + total + ")" : displayName;

        SwingUtilities.invokeLater(() -> showProgress(label, fileSize));

        try (Socket socket = new Socket(androidIp, FILE_TRANSFER_PORT);
             DataOutputStream dos = new DataOutputStream(socket.getOutputStream());
             FileInputStream fis = new FileInputStream(file)) {

            byte[] nameBytes = displayName.getBytes("UTF-8");
            dos.writeInt(nameBytes.length);
            dos.write(nameBytes);
            dos.writeLong(fileSize);
            dos.flush();

            byte[] buffer = new byte[65536];
            long sent = 0;
            int read;

            while ((read = fis.read(buffer)) != -1) {
                dos.write(buffer, 0, read);
                sent += read;
                final long finalSent = sent;
                SwingUtilities.invokeLater(() -> updateProgress(finalSent, fileSize));
            }

            dos.flush();
        } finally {
            SwingUtilities.invokeLater(FileDropZone::hideProgress);
        }
    }

    private static void showProgress(String fileName, long totalBytes) {
        if (progressDialog != null) return;

        progressDialog = new JDialog((Frame) null, "Sending File", false);
        progressDialog.setSize(360, 120);
        progressDialog.setLocationRelativeTo(null);
        progressDialog.setLayout(new BorderLayout());
        progressDialog.getContentPane().setBackground(Color.decode("#1E1E1E"));
        progressDialog.setDefaultCloseOperation(JDialog.DO_NOTHING_ON_CLOSE);

        JPanel inner = new JPanel();
        inner.setLayout(new BoxLayout(inner, BoxLayout.Y_AXIS));
        inner.setBackground(Color.decode("#1E1E1E"));
        inner.setBorder(BorderFactory.createEmptyBorder(16, 20, 16, 20));

        progressLabel = new JLabel("Sending: " + fileName);
        progressLabel.setForeground(Color.WHITE);
        progressLabel.setFont(new Font("Segoe UI", Font.PLAIN, 12));
        progressLabel.setAlignmentX(Component.LEFT_ALIGNMENT);

        progressBar = new JProgressBar(0, 100);
        progressBar.setValue(0);
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
    }

    private static void updateProgress(long sent, long total) {
        if (progressBar == null) return;
        int pct = (int) ((sent * 100) / total);
        progressBar.setValue(pct);
        progressBar.setString(pct + "%  (" + toMB(sent) + " / " + toMB(total) + ")");
    }

    private static void hideProgress() {
        if (progressDialog != null) {
            progressDialog.dispose();
            progressDialog = null;
            progressBar = null;
            progressLabel = null;
        }
    }

    private static String toMB(long bytes) {
        if (bytes < 1024) return bytes + " B";
        if (bytes < 1024 * 1024) return String.format("%.1f KB", bytes / 1024.0);
        return String.format("%.1f MB", bytes / (1024.0 * 1024));
    }
}