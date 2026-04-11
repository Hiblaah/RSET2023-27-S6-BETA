import javax.swing.*;
import java.awt.*;
import java.net.*;
import java.util.prefs.Preferences;

public class WindowsServer {

    static volatile String androidIp = null;
    static volatile boolean phoneConnected = false;
    static volatile boolean broadcasting = true;
    static JLabel batteryLabel;
    static JLabel modeLabel;

    static final int CLIPBOARD_PORT = 1234;
    static final int MIRROR_PORT = 3333;
    static final int DISCOVERY_PORT = 9876;
    static final String DISCOVERY_MESSAGE = "WINDROID_SERVER";
    private static final Preferences prefs = Preferences.userNodeForPackage(WindowsServer.class);
    private static final String PREF_PHONE_NAME = "paired_phone_name";
    private static final String PREF_PHONE_FINGERPRINT = "paired_phone_fingerprint";

    static JLabel subtitleLabel;
    static JLabel phoneNameLabel;
    static JLabel fingerprintLabel;
    static JPanel phonePanel;
    static JPanel noPhonePanel;
    static JToggleButton mirrorToggle;
    static JToggleButton clipboardToggle;
    static JToggleButton notifToggle;

    public static void main(String[] args) {
        System.out.println("=".repeat(60));
        System.out.println("Windroid PC Server v2.0");
        System.out.println("=".repeat(60));

        startDiscoveryBroadcast();
        ClipboardServer.start(CLIPBOARD_PORT);
        ScreenMirrorServer.start(MIRROR_PORT);
        FileReceiveServer.start();
        SwingUtilities.invokeLater(WindowsServer::createUI);
    }

    static void startDiscoveryBroadcast() {
        new Thread(() -> {
            try (DatagramSocket socket = new DatagramSocket()) {
                socket.setBroadcast(true);
                byte[] data = DISCOVERY_MESSAGE.getBytes();
                while (true) {
                    if (broadcasting) {
                        DatagramPacket packet = new DatagramPacket(
                                data, data.length,
                                InetAddress.getByName("255.255.255.255"),
                                DISCOVERY_PORT);
                        socket.send(packet);
                    }
                    Thread.sleep(1000);
                }
            } catch (Exception e) {
                System.err.println("Discovery error: " + e.getMessage());
            }
        }).start();
    }

    static void onPhoneConnected(String ip, String phoneName, String fingerprint) {
        androidIp = ip;
        phoneConnected = true;
        broadcasting = false;

        prefs.put(PREF_PHONE_NAME, phoneName);
        prefs.put(PREF_PHONE_FINGERPRINT, fingerprint);

        System.out.println("Phone paired: " + phoneName + " [" + fingerprint + "]");
        updateSubtitle("Connected: " + phoneName);
        SwingUtilities.invokeLater(() -> showPhonePanel(phoneName, fingerprint));
    }

    static void onPhoneDisconnected() {
        phoneConnected = false;
        androidIp = null;
        broadcasting = true; // ← add this
        System.out.println("Phone disconnected — waiting for reconnect");
        updateSubtitle("Phone disconnected — waiting...");
    }

    static void forgetPhone() {
        prefs.remove(PREF_PHONE_NAME);
        prefs.remove(PREF_PHONE_FINGERPRINT);
        androidIp = null;
        phoneConnected = false;
        broadcasting = true;
        System.out.println("Phone forgotten — resuming discovery");
        updateSubtitle("Waiting for Android connection...");
        SwingUtilities.invokeLater(() -> {
            phonePanel.setVisible(false);
            noPhonePanel.setVisible(true);
        });
    }

    static String getPairedFingerprint() {
        return prefs.get(PREF_PHONE_FINGERPRINT, null);
    }

    static String getPairedPhoneName() {
        return prefs.get(PREF_PHONE_NAME, null);
    }

    public static void updateSubtitle(String text) {
        SwingUtilities.invokeLater(() -> {
            if (subtitleLabel != null)
                subtitleLabel.setText(text);
        });
    }
public static void onMirrorStopped() {
    SwingUtilities.invokeLater(() -> {
        if (mirrorToggle != null && mirrorToggle.isSelected()) {
            mirrorToggle.setSelected(false);
            updateToggleStyle(mirrorToggle);
            // Also sync the ON/OFF dot — find it via the toggle's parent row
            Container row = mirrorToggle.getParent();
            if (row != null) {
                for (Component c : row.getComponents()) {
                    if (c instanceof JLabel label) {
                        label.setText("OFF");
                        label.setForeground(Color.decode("#555555"));
                    }
                }
            }
        }
        ScreenMirrorServer.setEnabled(false);
        System.out.println("Mirror toggle reset — permission was denied or service stopped");
    });
}
    private static void createUI() {
        try {
            UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName());
        } catch (Exception ignored) {
        }

        JFrame frame = new JFrame("Windroid");
        ImageIcon appIcon = new ImageIcon("mainlogo.png");
frame.setIconImage(appIcon.getImage());
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setSize(700, 420);
        frame.setMinimumSize(new Dimension(600, 360));
        frame.setLocationRelativeTo(null);
        frame.setLayout(new BorderLayout());
        frame.getContentPane().setBackground(Color.decode("#1A1A1A"));

        // ── Sidebar ──────────────────────────────────────────────────
        JPanel sidebar = new JPanel();
        sidebar.setLayout(new BoxLayout(sidebar, BoxLayout.Y_AXIS));
        sidebar.setBackground(Color.decode("#111111"));
        sidebar.setPreferredSize(new Dimension(220, 420));
        sidebar.setBorder(BorderFactory.createEmptyBorder(24, 18, 24, 18));

        JLabel sideHeader = new JLabel("PAIRED DEVICE");
        sideHeader.setForeground(Color.decode("#444444"));
        sideHeader.setFont(new Font("Segoe UI", Font.BOLD, 10));
        sideHeader.setAlignmentX(Component.LEFT_ALIGNMENT);
        sideHeader.setBorder(BorderFactory.createEmptyBorder(0, 0, 12, 0));

        // Phone panel
        phonePanel = new JPanel();
        phonePanel.setLayout(new BoxLayout(phonePanel, BoxLayout.Y_AXIS));
        phonePanel.setBackground(Color.decode("#111111"));
        phonePanel.setAlignmentX(Component.LEFT_ALIGNMENT);

        JPanel phoneCard = new JPanel();
        phoneCard.setLayout(new BoxLayout(phoneCard, BoxLayout.Y_AXIS));
        phoneCard.setBackground(Color.decode("#1E1E1E"));
        phoneCard.setBorder(BorderFactory.createCompoundBorder(
                BorderFactory.createLineBorder(Color.decode("#2A2A2A"), 1),
                BorderFactory.createEmptyBorder(12, 12, 12, 12)));
        phoneCard.setAlignmentX(Component.LEFT_ALIGNMENT);
        phoneCard.setMaximumSize(new Dimension(184, Integer.MAX_VALUE));

        JLabel deviceIcon = new JLabel("PHONE");
        deviceIcon.setForeground(Color.decode("#4CAF50"));
        deviceIcon.setFont(new Font("Segoe UI", Font.BOLD, 10));
        deviceIcon.setAlignmentX(Component.LEFT_ALIGNMENT);

        phoneNameLabel = new JLabel("—");
        phoneNameLabel.setForeground(Color.WHITE);
        phoneNameLabel.setFont(new Font("Segoe UI", Font.BOLD, 14));
        phoneNameLabel.setAlignmentX(Component.LEFT_ALIGNMENT);
        phoneNameLabel.setBorder(BorderFactory.createEmptyBorder(6, 0, 2, 0));

        fingerprintLabel = new JLabel("—");
        fingerprintLabel.setForeground(Color.decode("#444444"));
        fingerprintLabel.setFont(new Font("Segoe UI", Font.PLAIN, 9));
        fingerprintLabel.setAlignmentX(Component.LEFT_ALIGNMENT);

        batteryLabel = new JLabel("Battery: --%");
        batteryLabel.setForeground(Color.decode("#AAAAAA"));
        batteryLabel.setFont(new Font("Segoe UI", Font.PLAIN, 11));
        batteryLabel.setAlignmentX(Component.LEFT_ALIGNMENT);

        modeLabel = new JLabel("Mode: --");
        modeLabel.setForeground(Color.decode("#AAAAAA"));
        modeLabel.setFont(new Font("Segoe UI", Font.PLAIN, 11));
        modeLabel.setAlignmentX(Component.LEFT_ALIGNMENT);
        phoneCard.add(deviceIcon);
        phoneCard.add(phoneNameLabel);
        phoneCard.add(fingerprintLabel);

        phoneCard.add(Box.createVerticalStrut(6));
        phoneCard.add(batteryLabel);
        phoneCard.add(modeLabel);

        JButton forgetBtn = new JButton("Forget Device");
        forgetBtn.setBackground(Color.decode("#2A1A1A"));
        forgetBtn.setForeground(Color.decode("#FF5555"));
        forgetBtn.setBorderPainted(false);
        forgetBtn.setFocusPainted(false);
        forgetBtn.setFont(new Font("Segoe UI", Font.PLAIN, 11));
        forgetBtn.setAlignmentX(Component.LEFT_ALIGNMENT);
        forgetBtn.setMaximumSize(new Dimension(184, 30));
        forgetBtn.setCursor(Cursor.getPredefinedCursor(Cursor.HAND_CURSOR));
        forgetBtn.setBorder(BorderFactory.createEmptyBorder(6, 0, 0, 0));
        forgetBtn.setContentAreaFilled(false);
        forgetBtn.addActionListener(e -> {
            int confirm = JOptionPane.showConfirmDialog(
                    frame,
                    "Forget this device? You will need to re-pair from the Windroid app.",
                    "Forget Device",
                    JOptionPane.YES_NO_OPTION,
                    JOptionPane.WARNING_MESSAGE);
            if (confirm == JOptionPane.YES_OPTION)
                forgetPhone();
        });

        phonePanel.add(phoneCard);
        phonePanel.add(Box.createVerticalStrut(8));
        phonePanel.add(forgetBtn);

        // No phone panel
        noPhonePanel = new JPanel();
        noPhonePanel.setLayout(new BoxLayout(noPhonePanel, BoxLayout.Y_AXIS));
        noPhonePanel.setBackground(Color.decode("#111111"));
        noPhonePanel.setAlignmentX(Component.LEFT_ALIGNMENT);

        JPanel noPhoneCard = new JPanel();
        noPhoneCard.setLayout(new BoxLayout(noPhoneCard, BoxLayout.Y_AXIS));
        noPhoneCard.setBackground(Color.decode("#1A1A1A"));
        noPhoneCard.setBorder(BorderFactory.createCompoundBorder(
                BorderFactory.createLineBorder(Color.decode("#2A2A2A"), 1),
                BorderFactory.createEmptyBorder(12, 12, 12, 12)));
        noPhoneCard.setAlignmentX(Component.LEFT_ALIGNMENT);
        noPhoneCard.setMaximumSize(new Dimension(184, 80));

        JLabel noPhoneLabel = new JLabel("No device paired");
        noPhoneLabel.setForeground(Color.decode("#444444"));
        noPhoneLabel.setFont(new Font("Segoe UI", Font.PLAIN, 12));
        noPhoneLabel.setAlignmentX(Component.LEFT_ALIGNMENT);

        JLabel noPhoneHint = new JLabel("Open Windroid on phone");
        noPhoneHint.setForeground(Color.decode("#333333"));
        noPhoneHint.setFont(new Font("Segoe UI", Font.PLAIN, 10));
        noPhoneHint.setAlignmentX(Component.LEFT_ALIGNMENT);
        noPhoneHint.setBorder(BorderFactory.createEmptyBorder(4, 0, 0, 0));

        noPhoneCard.add(noPhoneLabel);
        noPhoneCard.add(noPhoneHint);
        noPhonePanel.add(noPhoneCard);

        String savedName = getPairedPhoneName();
        String savedFp = getPairedFingerprint();
        boolean hasPairedPhone = savedName != null && savedFp != null;

        if (hasPairedPhone) {
            phoneNameLabel.setText(savedName);
            fingerprintLabel.setText(savedFp);
            phonePanel.setVisible(true);
            noPhonePanel.setVisible(false);
            broadcasting = true;
        } else {
            phonePanel.setVisible(false);
            noPhonePanel.setVisible(true);
        }

        sidebar.add(sideHeader);
        sidebar.add(phonePanel);
        sidebar.add(noPhonePanel);

        // ── Main panel ───────────────────────────────────────────────
        JPanel main = new JPanel(new BorderLayout());
        main.setBackground(Color.decode("#1A1A1A"));

        // Header
        JPanel header = new JPanel(new BorderLayout());
        header.setBackground(Color.decode("#1A1A1A"));
        header.setBorder(BorderFactory.createEmptyBorder(28, 16, 20, 16));

        // LEFT SIDE (title + subtitle + status)
        JPanel leftHeader = new JPanel();
        leftHeader.setLayout(new BoxLayout(leftHeader, BoxLayout.Y_AXIS));
        leftHeader.setBackground(Color.decode("#1A1A1A"));
        leftHeader.setAlignmentX(Component.LEFT_ALIGNMENT);
        leftHeader.setMaximumSize(new Dimension(400, Integer.MAX_VALUE));

        ImageIcon logoIcon = new ImageIcon("logo.png"); // path to your image file
Image scaledLogo = logoIcon.getImage().getScaledInstance(90, 56, Image.SCALE_SMOOTH);

        JLabel logoLabel = new JLabel(new ImageIcon(scaledLogo));
        logoLabel.setBorder(BorderFactory.createEmptyBorder(0, 0, 0, 8));

        JPanel titleRow = new JPanel();
        titleRow.setLayout(new BoxLayout(titleRow, BoxLayout.X_AXIS));
        titleRow.setBackground(Color.decode("#1A1A1A"));
        titleRow.setAlignmentX(Component.LEFT_ALIGNMENT);
        titleRow.setMaximumSize(new Dimension(400, 60));
        logoLabel.setAlignmentY(Component.CENTER_ALIGNMENT);
        titleRow.add(logoLabel);
        

        JLabel titleLabel = new JLabel("Windroid");
        titleLabel.setForeground(Color.WHITE);
        titleLabel.setFont(new Font("Segoe UI", Font.BOLD, 22));
        titleLabel.setAlignmentY(Component.CENTER_ALIGNMENT);
        titleRow.add(titleLabel);
FileDropZone.attach(titleRow);


        subtitleLabel = new JLabel("Waiting for device...");
        subtitleLabel.setForeground(Color.decode("#555555"));
        subtitleLabel.setFont(new Font("Segoe UI", Font.PLAIN, 12));
        subtitleLabel.setAlignmentX(Component.LEFT_ALIGNMENT);

        JPanel statusRow = new JPanel();
        statusRow.setLayout(new BoxLayout(statusRow, BoxLayout.X_AXIS));
        statusRow.setBackground(Color.decode("#1A1A1A"));
        statusRow.setAlignmentX(Component.LEFT_ALIGNMENT);
        statusRow.setMaximumSize(new Dimension(400, 20));
        statusRow.add(batteryLabel);
        statusRow.add(Box.createHorizontalStrut(20));
        statusRow.add(modeLabel);

        leftHeader.add(titleRow);
        leftHeader.add(subtitleLabel);
        leftHeader.add(Box.createVerticalStrut(6));
        leftHeader.add(statusRow);

        // RIGHT SIDE (flashlight button)
        JToggleButton flashlightToggle = new JToggleButton("Flashlight");
        flashlightToggle.setFocusPainted(false);
        flashlightToggle.setFont(new Font("Segoe UI", Font.PLAIN, 12));
        flashlightToggle.setCursor(Cursor.getPredefinedCursor(Cursor.HAND_CURSOR));

        flashlightToggle.addActionListener(e -> {

            boolean on = flashlightToggle.isSelected();

            ClipboardServer.sendToAndroid(
                    on ? "CMD:FLASHLIGHT_ON"
                            : "CMD:FLASHLIGHT_OFF");
        });

        JPanel rightHeader = new JPanel();
        rightHeader.setBackground(Color.decode("#1A1A1A"));
        rightHeader.add(flashlightToggle);

        // ADD TO HEADER
        header.add(leftHeader, BorderLayout.WEST);
        header.add(rightHeader, BorderLayout.EAST);

        // Divider
        JSeparator sep = new JSeparator();
        sep.setForeground(Color.decode("#222222"));
        sep.setBackground(Color.decode("#222222"));

        // Features panel
        JPanel features = new JPanel();
        features.setLayout(new BoxLayout(features, BoxLayout.Y_AXIS));
        features.setBackground(Color.decode("#1A1A1A"));
        features.setBorder(BorderFactory.createEmptyBorder(20, 32, 20, 32));

        JLabel featuresHeader = new JLabel("FEATURES");
        featuresHeader.setForeground(Color.decode("#444444"));
        featuresHeader.setFont(new Font("Segoe UI", Font.BOLD, 10));
        featuresHeader.setAlignmentX(Component.LEFT_ALIGNMENT);
        featuresHeader.setBorder(BorderFactory.createEmptyBorder(0, 0, 14, 0));

        mirrorToggle = createFeatureToggle("Screen Mirror", "Stream your phone screen to PC", false);
        clipboardToggle = createFeatureToggle("Clipboard Sync", "Send clipboard content to PC", true);
        notifToggle = createFeatureToggle("Notifications", "Mirror phone notifications to PC", true);

        mirrorToggle.addActionListener(e -> {
            boolean on = mirrorToggle.isSelected();
            ScreenMirrorServer.setEnabled(on);
            updateToggleStyle(mirrorToggle);
            if (on) {
                ScreenMirrorServer.requestMirrorStart();
            } else {
                ScreenMirrorServer.requestMirrorStop();
            }
        });

        notifToggle.addActionListener(e -> {

            boolean on = notifToggle.isSelected();

            ClipboardServer.sendToAndroid(
                    on ? "CMD:FEATURE_NOTIFICATIONS_ON"
                            : "CMD:FEATURE_NOTIFICATIONS_OFF");
        });

        clipboardToggle.addActionListener(e -> {

            boolean on = clipboardToggle.isSelected();

            ClipboardServer.sendToAndroid(
                    on ? "CMD:FEATURE_CLIPBOARD_ON"
                            : "CMD:FEATURE_CLIPBOARD_OFF");
        });

        // Set initial enabled state
        ScreenMirrorServer.setEnabled(false);
        ClipboardServer.setClipboardEnabled(true);

        features.add(featuresHeader);
        features.add(createToggleRow(mirrorToggle));
        features.add(Box.createVerticalStrut(8));
        features.add(createToggleRow(clipboardToggle));
        features.add(Box.createVerticalStrut(8));
        features.add(createToggleRow(notifToggle));

        features.add(Box.createVerticalStrut(8));
        JPanel fileBrowseRow = new JPanel(new BorderLayout());
        fileBrowseRow.setBackground(Color.decode("#1E1E1E"));
        fileBrowseRow.setBorder(BorderFactory.createCompoundBorder(
                BorderFactory.createLineBorder(Color.decode("#2A2A2A"), 1),
                BorderFactory.createEmptyBorder(10, 14, 10, 14)));
        fileBrowseRow.setMaximumSize(new Dimension(Integer.MAX_VALUE, 50));

        JButton fileBrowseBtn = new JButton("Browse Android Files");
        fileBrowseBtn.setBackground(Color.decode("#1E1E1E"));
        fileBrowseBtn.setForeground(Color.decode("#ffffff"));
        fileBrowseBtn.setBorderPainted(false);
        fileBrowseBtn.setContentAreaFilled(false);
        fileBrowseBtn.setFocusPainted(false);
        fileBrowseBtn.setFont(new Font("Segoe UI", Font.PLAIN, 13));
        fileBrowseBtn.setHorizontalAlignment(SwingConstants.LEFT);
        fileBrowseBtn.setCursor(Cursor.getPredefinedCursor(Cursor.HAND_CURSOR));
        fileBrowseBtn.addActionListener(e -> AndroidFileBrowser.open());

        fileBrowseRow.add(fileBrowseBtn, BorderLayout.CENTER);
        features.add(fileBrowseRow);

        main.add(header, BorderLayout.NORTH);
        main.add(sep, BorderLayout.CENTER);
        main.add(features, BorderLayout.SOUTH);

        // Sidebar divider
        JSeparator sideDiv = new JSeparator(SwingConstants.VERTICAL);
        sideDiv.setForeground(Color.decode("#222222"));

        frame.add(sidebar, BorderLayout.WEST);
        frame.add(sideDiv, BorderLayout.LINE_START);
        frame.add(main, BorderLayout.CENTER);
        frame.setVisible(true);
    }

    private static JToggleButton createFeatureToggle(String title, String subtitle, boolean selected) {
        JToggleButton btn = new JToggleButton(title, selected);
        btn.setAlignmentX(Component.LEFT_ALIGNMENT);
        btn.setFont(new Font("Segoe UI", Font.PLAIN, 13));
        btn.setFocusPainted(false);
        btn.setCursor(Cursor.getPredefinedCursor(Cursor.HAND_CURSOR));
        btn.setHorizontalAlignment(SwingConstants.LEFT);
        updateToggleStyle(btn);
        return btn;
    }

    private static JPanel createToggleRow(JToggleButton toggle) {
        JPanel row = new JPanel(new BorderLayout());
        row.setBackground(Color.decode("#1E1E1E"));
        row.setBorder(BorderFactory.createCompoundBorder(
                BorderFactory.createLineBorder(Color.decode("#2A2A2A"), 1),
                BorderFactory.createEmptyBorder(10, 14, 10, 14)));
        row.setMaximumSize(new Dimension(Integer.MAX_VALUE, 50));

        toggle.setBackground(Color.decode("#1E1E1E"));
        toggle.setBorderPainted(false);
        toggle.setContentAreaFilled(false);

        JLabel statusDot = new JLabel(toggle.isSelected() ? "ON" : "OFF");
        statusDot.setForeground(toggle.isSelected() ? Color.decode("#4CAF50") : Color.decode("#555555"));
        statusDot.setFont(new Font("Segoe UI", Font.BOLD, 10));

        toggle.addActionListener(e -> {
            statusDot.setText(toggle.isSelected() ? "ON" : "OFF");
            statusDot.setForeground(toggle.isSelected() ? Color.decode("#4CAF50") : Color.decode("#555555"));
        });

        row.add(toggle, BorderLayout.CENTER);
        row.add(statusDot, BorderLayout.EAST);
        return row;
    }

    private static void updateToggleStyle(JToggleButton btn) {
        btn.setForeground(btn.isSelected() ? Color.WHITE : Color.decode("#666666"));
    }

    static void showPhonePanel(String name, String fingerprint) {
        phoneNameLabel.setText(name);
        fingerprintLabel.setText(fingerprint);
        phonePanel.setVisible(true);
        noPhonePanel.setVisible(false);
    }

    public static void sendInputToAndroid(String command) {
        // Routed through persistent connection — no separate socket
        System.out.println("Input command: " + command);
    }
}