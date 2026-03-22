import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Int16MultiArray


class NeckStateManager(Node):
    def __init__(self):
        super().__init__("neck_state_manager")

        self.pub = self.create_publisher(Int16MultiArray, "/robot/neck/cmd", 10)
        self.sub = self.create_subscription(
            String,
            "/robot/neck/events",
            self.on_event,
            10,
        )

        # ---------------- Base poses ----------------
        # Tune these later if needed.
        self.YAW_HOME = 385
        self.LIFT_HOME = 0
        self.PITCH_HOME = 0

        self.YAW_LEFT = 500
        self.YAW_RIGHT = 300

        self.LIFT_MID = 80
        self.LIFT_UP = 160

        self.PITCH_STRAIGHT = 0
        self.PITCH_UP = -40     # forward tilt
        self.PITCH_DOWN = 60    # back tilt

        # attention pose
        self.ATTN_YAW = 385
        self.ATTN_LIFT = 80
        self.ATTN_PITCH = 0

        # current desired semantic state
        self.current_yaw = self.YAW_HOME
        self.current_lift = self.LIFT_HOME
        self.current_pitch_offset = self.PITCH_HOME

        # nod action state
        self.nod_active = False
        self.nod_phase = 0
        self.nod_phase_started = 0.0
        self.nod_base_yaw = self.YAW_HOME
        self.nod_base_lift = self.LIFT_HOME
        self.nod_base_pitch = self.PITCH_HOME

        self.timer = self.create_timer(0.05, self.on_timer)

        self.publish_pose(self.current_yaw, self.current_lift, self.current_pitch_offset)
        self.get_logger().info("Neck state manager started")

    # ---------------- Publishing ----------------

    def publish_pose(self, yaw: int, lift: int, pitch_offset: int):
        msg = Int16MultiArray()
        msg.data = [int(yaw), int(lift), int(pitch_offset)]
        self.pub.publish(msg)

    def publish_home(self):
        msg = Int16MultiArray()
        msg.data = [-1, -1, -1]
        self.pub.publish(msg)

    # ---------------- Base semantic poses ----------------

    def set_pose(self, yaw: int, lift: int, pitch_offset: int):
        self.current_yaw = int(yaw)
        self.current_lift = int(lift)
        self.current_pitch_offset = int(pitch_offset)
        self.publish_pose(self.current_yaw, self.current_lift, self.current_pitch_offset)

    def set_home(self):
        self.current_yaw = self.YAW_HOME
        self.current_lift = self.LIFT_HOME
        self.current_pitch_offset = self.PITCH_HOME
        self.publish_home()

    def set_center(self):
        self.set_pose(self.YAW_HOME, self.current_lift, 0)

    def set_attention(self):
        self.set_pose(self.ATTN_YAW, self.ATTN_LIFT, self.ATTN_PITCH)

    def set_look_left(self):
        self.set_pose(self.YAW_LEFT, self.current_lift, self.current_pitch_offset)

    def set_look_right(self):
        self.set_pose(self.YAW_RIGHT, self.current_lift, self.current_pitch_offset)

    def set_look_up(self):
        # "up" here means visually lifting/straightening attention pose
        self.set_pose(self.current_yaw, self.LIFT_UP, -20)

    def set_look_down(self):
        # conservative down pose
        self.set_pose(self.current_yaw, self.LIFT_MID, 40)

    def set_lift_mid(self):
        self.set_pose(self.current_yaw, self.LIFT_MID, self.current_pitch_offset)

    def set_lift_up(self):
        self.set_pose(self.current_yaw, self.LIFT_UP, self.current_pitch_offset)

    # ---------------- Nod action ----------------

    def start_nod(self):
        if self.nod_active:
            return

        self.nod_active = True
        self.nod_phase = 0
        self.nod_phase_started = time.monotonic()

        self.nod_base_yaw = self.current_yaw
        self.nod_base_lift = self.current_lift
        self.nod_base_pitch = self.current_pitch_offset

        # first phase: slightly forward
        self.publish_pose(self.nod_base_yaw, self.nod_base_lift, self.nod_base_pitch - 35)

    def update_nod(self):
        if not self.nod_active:
            return

        now = time.monotonic()
        elapsed = now - self.nod_phase_started

        # phase 0: move forward
        if self.nod_phase == 0 and elapsed > 0.8:
            self.nod_phase = 1
            self.nod_phase_started = now
            self.publish_pose(self.nod_base_yaw, self.nod_base_lift, self.nod_base_pitch + 25)
            return

        # phase 1: move back
        if self.nod_phase == 1 and elapsed > 0.8:
            self.nod_phase = 2
            self.nod_phase_started = now
            self.publish_pose(self.nod_base_yaw, self.nod_base_lift, self.nod_base_pitch)
            return

        # phase 2: done
        if self.nod_phase == 2 and elapsed > 1.0:
            self.nod_active = False
            self.nod_phase = 0
            return

    # ---------------- Event handling ----------------

    def on_event(self, msg: String):
        event = msg.data.strip().lower()
        self.get_logger().info(f"neck event: {event}")

        if event == "home":
            self.nod_active = False
            self.set_home()

        elif event == "center":
            self.nod_active = False
            self.set_center()

        elif event == "attention":
            self.nod_active = False
            self.set_attention()

        elif event == "look_left":
            self.nod_active = False
            self.set_look_left()

        elif event == "look_right":
            self.nod_active = False
            self.set_look_right()

        elif event == "look_up":
            self.nod_active = False
            self.set_look_up()

        elif event == "look_down":
            self.nod_active = False
            self.set_look_down()

        elif event == "lift_mid":
            self.nod_active = False
            self.set_lift_mid()

        elif event == "lift_up":
            self.nod_active = False
            self.set_lift_up()

        elif event == "nod":
            self.start_nod()

        else:
            self.get_logger().warning(f"Unknown neck event: {event}")

    def on_timer(self):
        self.update_nod()


def main(args=None):
    rclpy.init(args=args)
    node = NeckStateManager()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
