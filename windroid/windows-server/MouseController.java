import java.awt.*;
import java.awt.event.InputEvent;

public class MouseController {

    private static Robot robot;

    static {
        try {
            robot = new Robot();
            robot.setAutoDelay(0);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static void move(int dx, int dy) {
        PointerInfo pointer = MouseInfo.getPointerInfo();
        if (pointer == null) return;
        Point p = pointer.getLocation();
        robot.mouseMove(p.x + dx, p.y + dy);
    }

    public static void click() {
        robot.mousePress(InputEvent.BUTTON1_DOWN_MASK);
        robot.mouseRelease(InputEvent.BUTTON1_DOWN_MASK);
    }

    public static void mouseDown() {
        robot.mousePress(InputEvent.BUTTON1_DOWN_MASK);
    }

    public static void mouseUp() {
        robot.mouseRelease(InputEvent.BUTTON1_DOWN_MASK);
    }

    public static void rightClick() {
        robot.mousePress(InputEvent.BUTTON3_DOWN_MASK);
        robot.mouseRelease(InputEvent.BUTTON3_DOWN_MASK);
    }

    public static void scroll(int amount) {
        robot.mouseWheel(amount / 10);
    }
}