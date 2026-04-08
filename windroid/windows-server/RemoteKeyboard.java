import java.awt.*;
import java.awt.event.KeyEvent;

public class RemoteKeyboard {

    private static Robot robot;

    static {
        try {
            robot = new Robot();
        } catch (AWTException e) {
            System.err.println("RemoteKeyboard: Failed to init Robot: " + e.getMessage());
        }
    }

    public static void handle(String message) {
        if (robot == null)
            return;
        if (!message.startsWith("KEY:"))
            return;

        String payload = message.substring(4);

        if (payload.startsWith("CHAR:")) {
            String ch = payload.substring(5);
            if (!ch.isEmpty())
                typeChar(ch.charAt(0));
            return;
        }

        switch (payload) {
            case "ENTER":
                pressKey(KeyEvent.VK_ENTER);
                break;
            case "BACKSPACE":
                pressKey(KeyEvent.VK_BACK_SPACE);
                break;
            case "TAB":
                pressKey(KeyEvent.VK_TAB);
                break;
            case "ESCAPE":
                pressKey(KeyEvent.VK_ESCAPE);
                break;
            case "UP":
                pressKey(KeyEvent.VK_UP);
                break;
            case "DOWN":
                pressKey(KeyEvent.VK_DOWN);
                break;
            case "LEFT":
                pressKey(KeyEvent.VK_LEFT);
                break;
            case "RIGHT":
                pressKey(KeyEvent.VK_RIGHT);
                break;
            case "SPACE":
                pressKey(KeyEvent.VK_SPACE);
                break;
            case "CTRL_C":
                pressCombo(KeyEvent.VK_CONTROL, KeyEvent.VK_C);
                break;
            case "CTRL_V":
                pressCombo(KeyEvent.VK_CONTROL, KeyEvent.VK_V);
                break;
            case "CTRL_Z":
                pressCombo(KeyEvent.VK_CONTROL, KeyEvent.VK_Z);
                break;
            case "CTRL_R":
                pressCombo(KeyEvent.VK_CONTROL, KeyEvent.VK_R);
                break;
            case "CTRL_S":
                pressCombo(KeyEvent.VK_CONTROL, KeyEvent.VK_S);
                break;
            case "CTRL_A":
                pressCombo(KeyEvent.VK_CONTROL, KeyEvent.VK_A);
                break;
            case "CTRL_X":
                pressCombo(KeyEvent.VK_CONTROL, KeyEvent.VK_X);
                break;
            case "CTRL_F":
                pressCombo(KeyEvent.VK_CONTROL, KeyEvent.VK_F);
                break;
            case "CTRL_W":
                pressCombo(KeyEvent.VK_CONTROL, KeyEvent.VK_W);
                break;
            case "CTRL_T":
                pressCombo(KeyEvent.VK_CONTROL, KeyEvent.VK_T);
                break;
            case "ALT_F4":
                pressCombo(KeyEvent.VK_ALT, KeyEvent.VK_F4);
                break;
            case "ALT_TAB":
                pressCombo(KeyEvent.VK_ALT, KeyEvent.VK_TAB);
                break;
            case "WIN_D":
                pressCombo(KeyEvent.VK_WINDOWS, KeyEvent.VK_D);
                break;
case "WIN_R":
    try {
        Runtime.getRuntime().exec(new String[]{
            "powershell", "-Command",
            "(New-Object -ComObject Shell.Application).FileRun()"
        });
    } catch (Exception e) {}
    break;
            case "WIN_E":
                pressCombo(KeyEvent.VK_WINDOWS, KeyEvent.VK_E);
                break;
            case "CTRL_SHIFT_T":
                pressTriple(KeyEvent.VK_CONTROL, KeyEvent.VK_SHIFT, KeyEvent.VK_T);
                break;
            case "CTRL_SHIFT_ESC":
                pressTriple(KeyEvent.VK_CONTROL, KeyEvent.VK_SHIFT, KeyEvent.VK_ESCAPE);
                break;
            case "TASK_MANAGER":
                try {
                    Runtime.getRuntime().exec("taskmgr.exe");
                } catch (Exception e) {
                }
                break;
        }
    }

    private static void pressKey(int keyCode) {
        try {
            robot.keyPress(keyCode);
            robot.keyRelease(keyCode);
        } catch (Exception e) {
            System.err.println("RemoteKeyboard: pressKey error: " + e.getMessage());
        }
    }

    private static void pressCombo(int modifier, int key) {
        try {
            robot.keyPress(modifier);
            robot.keyPress(key);
            robot.keyRelease(key);
            robot.keyRelease(modifier);
        } catch (Exception e) {
            System.err.println("RemoteKeyboard: pressCombo error: " + e.getMessage());
        }
    }

    private static void pressTriple(int mod1, int mod2, int key) {
        try {
            robot.keyPress(mod1);
            robot.keyPress(mod2);
            robot.keyPress(key);
            robot.keyRelease(key);
            robot.keyRelease(mod2);
            robot.keyRelease(mod1);
        } catch (Exception e) {
            System.err.println("RemoteKeyboard: pressTriple error: " + e.getMessage());
        }
    }

    private static void typeChar(char c) {
        try {
            boolean upper = Character.isUpperCase(c);
            boolean needShift = upper || isShiftChar(c);
            char base = upper ? Character.toLowerCase(c) : c;
            int keyCode = getKeyCode(base);

            if (keyCode == -1) {
                typeViaClipboard(String.valueOf(c));
                return;
            }

            if (needShift) {
                robot.keyPress(KeyEvent.VK_SHIFT);
            }
            robot.keyPress(keyCode);
            robot.keyRelease(keyCode);
            if (needShift) {
                robot.keyRelease(KeyEvent.VK_SHIFT);
            }
        } catch (Exception e) {
            typeViaClipboard(String.valueOf(c));
        }
    }

    private static void typeViaClipboard(String text) {
        try {
            java.awt.datatransfer.Clipboard clipboard = Toolkit.getDefaultToolkit().getSystemClipboard();
            java.awt.datatransfer.StringSelection sel = new java.awt.datatransfer.StringSelection(text);
            clipboard.setContents(sel, sel);
            robot.keyPress(KeyEvent.VK_CONTROL);
            robot.keyPress(KeyEvent.VK_V);
            robot.keyRelease(KeyEvent.VK_V);
            robot.keyRelease(KeyEvent.VK_CONTROL);
        } catch (Exception e) {
            System.err.println("RemoteKeyboard: clipboard type failed: " + e.getMessage());
        }
    }

    private static boolean isShiftChar(char c) {
        return "~!@#$%^&*()_+{}|:\"<>?".indexOf(c) >= 0;
    }

    private static int getKeyCode(char c) {
        if (c >= 'a' && c <= 'z')
            return KeyEvent.VK_A + (c - 'a');
        if (c >= '0' && c <= '9')
            return KeyEvent.VK_0 + (c - '0');
        switch (c) {
            case ' ':
                return KeyEvent.VK_SPACE;
            case '.':
                return KeyEvent.VK_PERIOD;
            case ',':
                return KeyEvent.VK_COMMA;
            case '!':
                return KeyEvent.VK_1;
            case '@':
                return KeyEvent.VK_2;
            case '#':
                return KeyEvent.VK_3;
            case '$':
                return KeyEvent.VK_4;
            case '%':
                return KeyEvent.VK_5;
            case '^':
                return KeyEvent.VK_6;
            case '&':
                return KeyEvent.VK_7;
            case '*':
                return KeyEvent.VK_8;
            case '(':
                return KeyEvent.VK_9;
            case ')':
                return KeyEvent.VK_0;
            case '-':
                return KeyEvent.VK_MINUS;
            case '_':
                return KeyEvent.VK_MINUS;
            case '=':
                return KeyEvent.VK_EQUALS;
            case '+':
                return KeyEvent.VK_EQUALS;
            case '[':
                return KeyEvent.VK_OPEN_BRACKET;
            case ']':
                return KeyEvent.VK_CLOSE_BRACKET;
            case '\\':
                return KeyEvent.VK_BACK_SLASH;
            case ';':
                return KeyEvent.VK_SEMICOLON;
            case '\'':
                return KeyEvent.VK_QUOTE;
            case '/':
                return KeyEvent.VK_SLASH;
            case '`':
                return KeyEvent.VK_BACK_QUOTE;
            default:
                return -1;
        }
    }
}