import javax.swing.*;
import java.awt.*;
import java.util.Queue;
import java.util.concurrent.ConcurrentLinkedQueue;

public class NotificationServer {

    private static volatile boolean enabled = true;
    private static final Queue<JWindow> activeToasts = new ConcurrentLinkedQueue<>();

    private static final int TOAST_WIDTH = 380;
    private static final int TOAST_MARGIN = 12;

    public static void setEnabled(boolean val) { enabled = val; }

    public static void handle(String line) {
        if (!enabled) return;

        int lastPipe = line.lastIndexOf('|');
        if (lastPipe < 0) return;

        String payload = line.substring(lastPipe + 1);
        String[] imageParts = payload.split(",", 2);

        String iconBase64 = imageParts.length > 0 ? imageParts[0] : "";
        String imageBase64 = imageParts.length > 1 ? imageParts[1] : "";

        String[] parts = line.substring(0, lastPipe).split("\\|", 7);

        String appName    = parts.length > 1 ? parts[1] : "Unknown";
        String chatTitle  = parts.length > 2 ? parts[2] : "";
        String senderName = parts.length > 3 ? parts[3] : "";
        String text       = parts.length > 4 ? parts[4] : "";

        show(appName, chatTitle, senderName, text, iconBase64, imageBase64);
    }

    private static void show(String appName, String chatTitle, String senderName,
                             final String text, String iconBase64, String imageBase64) {

        SwingUtilities.invokeLater(() -> {

            JWindow toast = new JWindow();
            toast.setAlwaysOnTop(true);

            boolean isGroup = !chatTitle.isEmpty() && !senderName.isEmpty()
                    && !chatTitle.equals(senderName);

            JPanel panel = new JPanel(new BorderLayout(12, 8));
            panel.setBackground(Color.decode("#1F1F1F"));
            panel.setBorder(BorderFactory.createCompoundBorder(
                    BorderFactory.createLineBorder(Color.decode("#3A3A3A")),
                    BorderFactory.createEmptyBorder(12, 14, 12, 14)
            ));

            JPanel topBar = new JPanel(new BorderLayout());
            topBar.setOpaque(false);

            JLabel appLabel = new JLabel(appName);
            appLabel.setForeground(Color.decode("#AAAAAA"));
            appLabel.setFont(new Font("Segoe UI", Font.PLAIN, 11));

            JLabel close = new JLabel("✕");
            close.setForeground(Color.decode("#AAAAAA"));
            close.setFont(new Font("Segoe UI", Font.BOLD, 14));
            close.setCursor(Cursor.getPredefinedCursor(Cursor.HAND_CURSOR));
            close.addMouseListener(new java.awt.event.MouseAdapter() {
                public void mouseClicked(java.awt.event.MouseEvent e) {
                    toast.dispose();
                    activeToasts.remove(toast);
                    reposition();
                }
            });

            topBar.add(appLabel, BorderLayout.WEST);
            topBar.add(close, BorderLayout.EAST);

            panel.add(topBar, BorderLayout.NORTH);

            JLabel iconLabel = new JLabel();
            iconLabel.setPreferredSize(new Dimension(48, 48));

            if (!iconBase64.isEmpty()) {
                try {
                    byte[] imageBytes = java.util.Base64.getDecoder().decode(iconBase64);
                    Image scaled = new ImageIcon(imageBytes).getImage()
                            .getScaledInstance(48, 48, Image.SCALE_SMOOTH);
                    iconLabel.setIcon(new ImageIcon(scaled));
                } catch (Exception ignored) {}
            }

            panel.add(iconLabel, BorderLayout.WEST);

            JPanel textPanel = new JPanel();
            textPanel.setLayout(new BoxLayout(textPanel, BoxLayout.Y_AXIS));
            textPanel.setOpaque(false);

            if (isGroup) {
                JLabel group = new JLabel(chatTitle);
                group.setForeground(Color.decode("#888888"));
                group.setFont(new Font("Segoe UI", Font.PLAIN, 11));
                textPanel.add(group);
            }

            String displayName = !senderName.isEmpty() ? senderName
                    : !chatTitle.isEmpty() ? chatTitle
                    : appName;

            JLabel nameLabel = new JLabel(displayName);
            nameLabel.setForeground(Color.WHITE);
            nameLabel.setFont(new Font("Segoe UI", Font.BOLD, 13));

            final String safeText = text.replace("<","&lt;").replace(">","&gt;");

            JLabel msgLabel = new JLabel(
                    "<html><div style='width:240px'>" + safeText + "</div></html>");
            msgLabel.setForeground(Color.decode("#CCCCCC"));
            msgLabel.setFont(new Font("Segoe UI", Font.PLAIN, 12));

            textPanel.add(nameLabel);
            textPanel.add(Box.createVerticalStrut(3));
            textPanel.add(msgLabel);

            if (!imageBase64.isEmpty()) {
                try {
                    byte[] imgBytes = java.util.Base64.getDecoder().decode(imageBase64);
                    Image img = new ImageIcon(imgBytes).getImage()
                            .getScaledInstance(220, 120, Image.SCALE_SMOOTH);

                    JLabel imgLabel = new JLabel(new ImageIcon(img));
                    imgLabel.setBorder(BorderFactory.createEmptyBorder(6,0,0,0));
                    textPanel.add(imgLabel);

                } catch (Exception ignored) {}
            }

            panel.add(textPanel, BorderLayout.CENTER);

            toast.add(panel);

            panel.addMouseListener(new java.awt.event.MouseAdapter() {
                public void mouseEntered(java.awt.event.MouseEvent e) {
                    panel.setBackground(Color.decode("#262626"));
                }
                public void mouseExited(java.awt.event.MouseEvent e) {
                    panel.setBackground(Color.decode("#1F1F1F"));
                }
            });

            toast.setSize(TOAST_WIDTH, panel.getPreferredSize().height + 10);

            activeToasts.add(toast);
            reposition();

            toast.setVisible(true);

            Timer timer = new Timer(6000, e -> {
                toast.dispose();
                activeToasts.remove(toast);
                reposition();
            });
            timer.setRepeats(false);
            timer.start();
        });
    }

    private static void reposition() {

        Dimension screen = Toolkit.getDefaultToolkit().getScreenSize();

        int y = screen.height - 50;

        for (JWindow toast : activeToasts) {

            int height = toast.getHeight();
            y -= height + TOAST_MARGIN;

            int x = screen.width - TOAST_WIDTH - TOAST_MARGIN;

            toast.setLocation(x, y);
        }
    }
}