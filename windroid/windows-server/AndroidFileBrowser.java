import javax.swing.*;
import javax.swing.border.*;
import java.awt.*;
import java.awt.event.*;
import java.io.File;
import java.io.FileInputStream;

class FileAccessHandler {

    public static void handleMessage(String message) {
        if (message.equals("FILE_REQ_DRIVES")) {
            StringBuilder response = new StringBuilder("FILE_RES_LIST|");
            for (File root : File.listRoots()) {
                String drive = root.getAbsolutePath().replace("\\", "");
                response.append(drive).append(",DIR;");
            }
            ClipboardServer.sendToAndroid(response.toString());
            return;
        }

        System.out.println("FileAccessHandler received: " + message);

        try {
            if (message.startsWith("FILE_REQ_LIST|")) {
                String path = message.substring("FILE_REQ_LIST|".length());
                File dir = new File(path);

                if (!dir.exists() || !dir.isDirectory()) return;

                StringBuilder response = new StringBuilder("FILE_RES_LIST|");
                File[] files = dir.listFiles();
                if (files != null) {
                    java.util.Arrays.sort(files, (a, b) -> {
                        if (a.isDirectory() && !b.isDirectory()) return -1;
                        if (!a.isDirectory() && b.isDirectory()) return 1;
                        return a.getName().compareToIgnoreCase(b.getName());
                    });
                    for (File f : files) {
                        try {
                            response.append(f.getName())
                                    .append(",")
                                    .append(f.isDirectory() ? "DIR" : "FILE")
                                    .append(";");
                        } catch (Exception ignored) {}
                    }
                }
                ClipboardServer.sendToAndroid(response.toString());
            }

            else if (message.startsWith("FILE_REQ_DOWNLOAD|")) {
                String path = message.substring("FILE_REQ_DOWNLOAD|".length());
                File file = new File(path);

                if (!file.exists() || !file.isFile()) return;

                System.out.println("Sending file to Android: " + file.getName());

                byte[] data = new byte[(int) file.length()];
                try (FileInputStream fis = new FileInputStream(file)) {
                    fis.read(data);
                }

                String base64 = java.util.Base64.getEncoder().encodeToString(data);
                ClipboardServer.sendToAndroid("FILE_RES_FILE|" + file.getName() + "|" + base64);
            }

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}

public class AndroidFileBrowser {

    private static final Color BG       = Color.decode("#1A1A1A");
    private static final Color BG2      = Color.decode("#222222");
    private static final Color BG3      = Color.decode("#2A2A2A");
    private static final Color FG       = Color.decode("#E0E0E0");
    private static final Color FG_DIM   = Color.decode("#666666");
    private static final Color HOVER    = Color.decode("#2D2D2D");

    private static JFrame frame;
    private static DefaultListModel<String> model;
    private static JList<String> list;
    private static JLabel pathLabel;
    private static JLabel itemCountLabel;
    private static String currentPath = "/storage/emulated/0";
    private static String downloadPath = System.getProperty("user.home") + "\\Downloads";

    public static void open() {
        if (frame == null) {
            frame = new JFrame("Android Files");
            frame.setSize(520, 620);
            frame.setLocationRelativeTo(null);
            frame.setLayout(new BorderLayout());
            frame.getContentPane().setBackground(BG);

            model = new DefaultListModel<>();
            list = new JList<>(model);
            list.setBackground(BG2);
            list.setForeground(FG);
            list.setSelectionBackground(BG3);
            list.setSelectionForeground(Color.WHITE);
            list.setFont(new Font("Segoe UI", Font.PLAIN, 13));
            list.setFixedCellHeight(36);
            list.setBorder(BorderFactory.createEmptyBorder(4, 8, 4, 8));

            list.setCellRenderer(new DefaultListCellRenderer() {
                @Override
                public Component getListCellRendererComponent(JList<?> l, Object value,
                        int index, boolean isSelected, boolean cellHasFocus) {
                    JLabel lbl = (JLabel) super.getListCellRendererComponent(
                            l, value, index, isSelected, cellHasFocus);
                    String text = value.toString();
                    lbl.setBackground(isSelected ? BG3 : (index % 2 == 0 ? BG2 : Color.decode("#252525")));
                    lbl.setForeground(text.endsWith("/") ? Color.WHITE : Color.decode("#AAAAAA"));
                    lbl.setBorder(BorderFactory.createEmptyBorder(0, 8, 0, 8));
                    return lbl;
                }
            });
JButton backButton = makeIconButton("<", "Back");
            JButton refreshButton = makeIconButton("Refresh", "Refresh");
            JButton settingsButton = makeIconButton("Settings", "Settings");

            backButton.addActionListener(e -> goBack());
            refreshButton.addActionListener(e -> requestFolder(currentPath));
            settingsButton.addActionListener(e -> openSettings(frame));

            pathLabel = new JLabel(currentPath);
            pathLabel.setForeground(FG_DIM);
            pathLabel.setFont(new Font("Segoe UI", Font.PLAIN, 11));

            itemCountLabel = new JLabel("");
            itemCountLabel.setForeground(FG_DIM);
            itemCountLabel.setFont(new Font("Segoe UI", Font.PLAIN, 11));

            JPanel pathRow = new JPanel(new BorderLayout(8, 0));
            pathRow.setBackground(BG);
            pathRow.setBorder(BorderFactory.createEmptyBorder(5, 12, 5, 12));
            pathRow.add(pathLabel, BorderLayout.CENTER);
            pathRow.add(itemCountLabel, BorderLayout.EAST);

            JPanel btnRow = new JPanel(new BorderLayout(4, 0));
            btnRow.setBackground(Color.decode("#111111"));
            btnRow.setBorder(BorderFactory.createEmptyBorder(8, 12, 8, 12));

            JPanel leftBtns = new JPanel(new FlowLayout(FlowLayout.LEFT, 4, 0));
            leftBtns.setBackground(Color.decode("#111111"));
            leftBtns.add(backButton);
            leftBtns.add(refreshButton);

            btnRow.add(leftBtns, BorderLayout.WEST);
            btnRow.add(settingsButton, BorderLayout.EAST);

            JSeparator sep = new JSeparator();
            sep.setForeground(BG3);
            sep.setBackground(BG3);

            JPanel topPanel = new JPanel(new BorderLayout());
            topPanel.setBackground(BG);
            topPanel.add(btnRow, BorderLayout.NORTH);
            topPanel.add(sep, BorderLayout.CENTER);
            topPanel.add(pathRow, BorderLayout.SOUTH);

            JScrollPane scrollPane = new JScrollPane(list);
            scrollPane.setBackground(BG2);
            scrollPane.getViewport().setBackground(BG2);
            scrollPane.setBorder(BorderFactory.createEmptyBorder());

            JPanel statusBar = new JPanel(new BorderLayout());
            statusBar.setBackground(Color.decode("#111111"));
            statusBar.setBorder(BorderFactory.createEmptyBorder(6, 12, 6, 12));
            JLabel statusLabel = new JLabel("Double-click folder to open  •  Right-click file to download");
            statusLabel.setForeground(FG_DIM);
            statusLabel.setFont(new Font("Segoe UI", Font.PLAIN, 10));
            statusBar.add(statusLabel, BorderLayout.WEST);

            list.addMouseListener(new MouseAdapter() {
                public void mouseClicked(MouseEvent evt) {
                    if (evt.getClickCount() == 2) {
                        String item = list.getSelectedValue();
                        if (item == null) return;
                        if (item.endsWith("/")) {
                            currentPath = currentPath + "/" + item.replace("/", "");
                            requestFolder(currentPath);
                        }
                    }
                }

                public void mousePressed(MouseEvent evt) {
                    if (SwingUtilities.isRightMouseButton(evt)) {
                        int index = list.locationToIndex(evt.getPoint());
                        if (index < 0) return;
                        list.setSelectedIndex(index);
                        String item = list.getSelectedValue();
                        if (item == null || item.endsWith("/")) return;

                        JPopupMenu menu = new JPopupMenu();
                        menu.setBackground(BG2);
                        menu.setBorder(BorderFactory.createLineBorder(BG3));

                        JMenuItem downloadItem = new JMenuItem("⬇  Download");
                        downloadItem.setBackground(BG2);
                        downloadItem.setForeground(FG);
                        downloadItem.setFont(new Font("Segoe UI", Font.PLAIN, 12));
                        downloadItem.setBorder(BorderFactory.createEmptyBorder(6, 12, 6, 12));
                        downloadItem.addActionListener(e ->
                            ClipboardServer.sendToAndroid(
                                "ANDROID_REQ_DOWNLOAD|" + currentPath + "/" + item)
                        );

                        menu.add(downloadItem);
                        menu.show(list, evt.getX(), evt.getY());
                    }
                }
            });

            frame.add(topPanel, BorderLayout.NORTH);
            frame.add(scrollPane, BorderLayout.CENTER);
            frame.add(statusBar, BorderLayout.SOUTH);
        }

        frame.setVisible(true);
        requestFolder(currentPath);
    }
private static JButton makeIconButton(String text, String tooltip) {
    JButton btn = new JButton(text);
    btn.setToolTipText(tooltip);

    btn.setBackground(BG3);
    btn.setForeground(FG);
    btn.setFocusPainted(false);
    btn.setBorderPainted(false);

    btn.setFont(new Font("Segoe UI", Font.PLAIN, 13));
    btn.setCursor(Cursor.getPredefinedCursor(Cursor.HAND_CURSOR));

    // padding instead of fixed size
    btn.setBorder(BorderFactory.createEmptyBorder(4, 12, 4, 12));

    btn.addMouseListener(new MouseAdapter() {
        public void mouseEntered(MouseEvent e) { btn.setBackground(HOVER); }
        public void mouseExited(MouseEvent e)  { btn.setBackground(BG3); }
    });

    return btn;
}

    public static void openSettings(Component parent) {
        JPanel panel = new JPanel(new BorderLayout(8, 8));
        panel.setBackground(BG2);
        panel.setBorder(BorderFactory.createEmptyBorder(12, 12, 12, 12));

        JLabel label = new JLabel("Download location:");
        label.setForeground(FG);
        label.setFont(new Font("Segoe UI", Font.PLAIN, 12));

        JTextField pathField = new JTextField(downloadPath, 30);
        pathField.setBackground(BG3);
        pathField.setForeground(FG);
        pathField.setCaretColor(FG);
        pathField.setBorder(BorderFactory.createCompoundBorder(
            BorderFactory.createLineBorder(Color.decode("#333333")),
            BorderFactory.createEmptyBorder(4, 8, 4, 8)
        ));

        JButton browseBtn = new JButton("Browse...");
        browseBtn.setBackground(BG3);
        browseBtn.setForeground(FG);
        browseBtn.setFocusPainted(false);
        browseBtn.setBorderPainted(false);
        browseBtn.addActionListener(e -> {
            JFileChooser chooser = new JFileChooser(downloadPath);
            chooser.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);
            if (chooser.showOpenDialog(parent) == JFileChooser.APPROVE_OPTION)
                pathField.setText(chooser.getSelectedFile().getAbsolutePath());
        });

        JPanel row = new JPanel(new BorderLayout(4, 0));
        row.setBackground(BG2);
        row.add(pathField, BorderLayout.CENTER);
        row.add(browseBtn, BorderLayout.EAST);

        panel.add(label, BorderLayout.NORTH);
        panel.add(row, BorderLayout.CENTER);

        int result = JOptionPane.showConfirmDialog(
            parent, panel, "File Browser Settings",
            JOptionPane.OK_CANCEL_OPTION, JOptionPane.PLAIN_MESSAGE
        );

        if (result == JOptionPane.OK_OPTION)
            downloadPath = pathField.getText().trim();
    }

    private static void goBack() {
        if (currentPath.equals("/storage/emulated/0")) return;
        int lastSlash = currentPath.lastIndexOf("/");
        currentPath = currentPath.substring(0, lastSlash);
        requestFolder(currentPath);
    }

    private static void requestFolder(String path) {
        if (pathLabel != null) pathLabel.setText(path.isEmpty() ? "Drives" : path);
        ClipboardServer.sendToAndroid("ANDROID_REQ_LIST|" + path);
    }

    public static String getDownloadPath() {
        return downloadPath;
    }

    public static void updateList(String message) {
        SwingUtilities.invokeLater(() -> {
            model.clear();
            String data = message.replace("ANDROID_RES_LIST|", "");
            String[] items = data.split(";");
            int count = 0;

            for (String item : items) {
                if (item.isEmpty()) continue;
                String[] parts = item.split(",", 2);
                if (parts.length < 2) continue;
                String name = parts[0];
                String type = parts[1];
                model.addElement(type.equals("DIR") ? name + "/" : name);
                count++;
            }

            final int total = count;
            if (itemCountLabel != null)
                itemCountLabel.setText(total + " items");
        });
    }
}