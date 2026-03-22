import rclpy
from rclpy.node import Node
from std_msgs.msg import String, UInt8MultiArray


HEAD_OFF = 0
HEAD_IDLE = 1
HEAD_BOOT = 2
HEAD_LISTENING = 3
HEAD_THINKING = 4
HEAD_SPEAKING = 5
HEAD_CHARGING = 6
HEAD_SUCCESS = 7
HEAD_WARNING = 8
HEAD_ERROR = 9

SPEAK_STYLE_CALM = 0
SPEAK_STYLE_ACTIVE = 1


class HeadStateManager(Node):
    def __init__(self):
        super().__init__("head_state_manager")

        self.pub = self.create_publisher(UInt8MultiArray, "/robot/head/cmd", 10)
        self.sub = self.create_subscription(String, "/robot/head/events", self.on_event, 10)

        self.default_brightness = self.declare_parameter("default_brightness", 64).value
        self.default_style = self.declare_parameter("default_speaking_style", SPEAK_STYLE_CALM).value

        self.boot_active = True
        self.listening_active = False
        self.thinking_active = False
        self.speaking_active = False
        self.charging_active = False
        self.warning_active = False
        self.error_active = False

        self.speaking_style = self.default_style
        self.speaking_level = 100

        self.success_timer = None

        self.publish_current_state()
        self.get_logger().info("Head state manager started")

    def on_event(self, msg: String) -> None:
        event = msg.data.strip()
        self.get_logger().info(f"Head event: {event}")

        if event == "boot_started":
            self.boot_active = True

        elif event == "system_ready":
            self.boot_active = False

        elif event == "listening_started":
            self.listening_active = True
            self.thinking_active = False

        elif event == "listening_stopped":
            self.listening_active = False

        elif event == "thinking_started":
            self.thinking_active = True
            self.listening_active = False

        elif event == "thinking_stopped":
            self.thinking_active = False

        elif event == "speaking_started":
            self.speaking_active = True
            self.thinking_active = False
            self.listening_active = False

        elif event == "speaking_finished":
            self.speaking_active = False
            self.speaking_level = 0

        elif event == "charging_started":
            self.charging_active = True

        elif event == "charging_stopped":
            self.charging_active = False

        elif event == "warning_on":
            self.warning_active = True

        elif event == "warning_off":
            self.warning_active = False

        elif event == "error_on":
            self.error_active = True

        elif event == "error_off":
            self.error_active = False

        elif event == "success":
            self.trigger_success()
            return

        elif event == "speak_style_calm":
            self.speaking_style = SPEAK_STYLE_CALM

        elif event == "speak_style_active":
            self.speaking_style = SPEAK_STYLE_ACTIVE

        else:
            self.get_logger().warn(f"Unknown head event: {event}")

        self.publish_current_state()

    def trigger_success(self) -> None:
        self.publish_cmd(HEAD_SUCCESS, self.default_brightness, 0, self.speaking_style)

        if self.success_timer is not None:
            self.success_timer.cancel()

        self.success_timer = self.create_timer(1.0, self.on_success_timeout)

    def on_success_timeout(self) -> None:
        if self.success_timer is not None:
            self.success_timer.cancel()
            self.success_timer = None
        self.publish_current_state()

    def publish_current_state(self) -> None:
        if self.error_active:
            self.publish_cmd(HEAD_ERROR, self.default_brightness, 0, self.speaking_style)
            return

        if self.warning_active:
            self.publish_cmd(HEAD_WARNING, self.default_brightness, 0, self.speaking_style)
            return

        if self.charging_active:
            self.publish_cmd(HEAD_CHARGING, self.default_brightness, 0, self.speaking_style)
            return

        if self.speaking_active:
            self.publish_cmd(HEAD_SPEAKING, self.default_brightness, self.speaking_level, self.speaking_style)
            return

        if self.thinking_active:
            self.publish_cmd(HEAD_THINKING, self.default_brightness, 0, self.speaking_style)
            return

        if self.listening_active:
            self.publish_cmd(HEAD_LISTENING, self.default_brightness, 0, self.speaking_style)
            return

        if self.boot_active:
            self.publish_cmd(HEAD_BOOT, self.default_brightness, 0, self.speaking_style)
            return

        self.publish_cmd(HEAD_IDLE, self.default_brightness, 0, self.speaking_style)

    def publish_cmd(self, mode: int, brightness: int, level: int, style: int) -> None:
        msg = UInt8MultiArray()
        msg.data = [mode, brightness, level, style]
        self.pub.publish(msg)

    def destroy_node(self):
        if self.success_timer is not None:
            self.success_timer.cancel()
            self.success_timer = None
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = HeadStateManager()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
