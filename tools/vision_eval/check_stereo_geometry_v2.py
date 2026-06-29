#!/usr/bin/env python3
from pathlib import Path

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


class StereoGeometryDiagnostic(Node):
    def __init__(self):
        super().__init__("stereo_geometry_diagnostic")

        self.bridge = CvBridge()
        self.left = {}
        self.right = {}
        self.done = False

        self.output_dir = (
            Path.home()
            / "ai_robot"
            / "data"
            / "stereo_diagnostics"
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.create_subscription(
            Image,
            "/camera/left/image_rect",
            self.left_cb,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Image,
            "/camera/right/image_rect",
            self.right_cb,
            qos_profile_sensor_data,
        )

        self.get_logger().info(
            "Waiting for an exact-timestamp rectified stereo pair..."
        )

    @staticmethod
    def key(msg):
        return int(msg.header.stamp.sec), int(msg.header.stamp.nanosec)

    def left_cb(self, msg):
        if self.done:
            return
        self.left[self.key(msg)] = msg
        self.try_pair()
        self.trim()

    def right_cb(self, msg):
        if self.done:
            return
        self.right[self.key(msg)] = msg
        self.try_pair()
        self.trim()

    def trim(self):
        for store in (self.left, self.right):
            if len(store) > 40:
                for key in sorted(store)[:-20]:
                    store.pop(key, None)

    def try_pair(self):
        common = sorted(set(self.left).intersection(self.right))
        if not common or self.done:
            return

        stamp = common[-1]
        self.done = True

        left = self.bridge.imgmsg_to_cv2(
            self.left[stamp],
            desired_encoding="bgr8",
        )
        right = self.bridge.imgmsg_to_cv2(
            self.right[stamp],
            desired_encoding="bgr8",
        )

        self.analyse(left, right, stamp)

    @staticmethod
    def summary(name, values):
        if not values:
            print(f"{name}: no values")
            return
        arr = np.asarray(values, dtype=np.float32)
        print(
            f"{name}: "
            f"median={np.median(arr):.2f}px, "
            f"p10={np.percentile(arr, 10):.2f}px, "
            f"p90={np.percentile(arr, 90):.2f}px, "
            f"min={arr.min():.2f}px, "
            f"max={arr.max():.2f}px"
        )

    def analyse(self, left, right, stamp):
        gray_left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

        orb = cv2.ORB_create(nfeatures=4000)
        kp_l, des_l = orb.detectAndCompute(gray_left, None)
        kp_r, des_r = orb.detectAndCompute(gray_right, None)

        if des_l is None or des_r is None:
            print("Not enough descriptors.")
            self.finish()
            return

        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        knn = matcher.knnMatch(des_l, des_r, k=2)

        good = []
        for pair in knn:
            if len(pair) != 2:
                continue
            m, n = pair
            if m.distance < 0.75 * n.distance:
                good.append(m)

        dx = []
        dy_signed = []
        dy_abs = []
        normal = []
        swapped = []

        for match in good:
            xl, yl = kp_l[match.queryIdx].pt
            xr, yr = kp_r[match.trainIdx].pt

            delta_x = xl - xr
            delta_y = yl - yr

            dx.append(delta_x)
            dy_signed.append(delta_y)
            dy_abs.append(abs(delta_y))

            if abs(delta_y) <= 5.0 and 0.0 < delta_x <= 512.0:
                normal.append(match)

            if abs(delta_y) <= 5.0 and -512.0 <= delta_x < 0.0:
                swapped.append(match)

        print(f"Exact stamp: {stamp[0]}.{stamp[1]:09d}")
        print(f"Image size: {left.shape[1]}x{left.shape[0]}")
        print(f"Left keypoints: {len(kp_l)}")
        print(f"Right keypoints: {len(kp_r)}")
        print(f"Ratio-test matches: {len(good)}")

        self.summary("Signed dx = x_left - x_right", dx)
        self.summary("Signed dy = y_left - y_right", dy_signed)
        self.summary("Absolute vertical error |dy|", dy_abs)

        print(
            f"Rectified normal-sign matches "
            f"(|dy|<=5, dx>0): {len(normal)}"
        )
        print(
            f"Rectified reversed-sign matches "
            f"(|dy|<=5, dx<0): {len(swapped)}"
        )

        if len(normal) >= max(10, len(swapped) * 2):
            verdict = "NORMAL_ORDER_LIKELY"
        elif len(swapped) >= max(10, len(normal) * 2):
            verdict = "LEFT_RIGHT_TOPICS_OR_CALIBRATION_ORDER_LIKELY_SWAPPED"
        elif dy_abs and float(np.median(dy_abs)) > 5.0:
            verdict = "RECTIFICATION_VERTICAL_ALIGNMENT_BAD"
        else:
            verdict = "AMBIGUOUS_OR_SCENE_NOT_SUITABLE"

        print(f"Verdict: {verdict}")

        stamp_text = f"{stamp[0]}_{stamp[1]:09d}"

        all_matches_path = (
            self.output_dir
            / f"geometry_all_matches_{stamp_text}.png"
        )
        best_all = sorted(good, key=lambda m: m.distance)[:150]
        all_image = cv2.drawMatches(
            left,
            kp_l,
            right,
            kp_r,
            best_all,
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        cv2.imwrite(str(all_matches_path), all_image)

        horizontal_path = (
            self.output_dir
            / f"geometry_horizontal_lines_{stamp_text}.png"
        )
        left_lines = left.copy()
        right_lines = right.copy()
        for y in range(0, left.shape[0], 40):
            cv2.line(
                left_lines,
                (0, y),
                (left.shape[1] - 1, y),
                (0, 255, 0),
                1,
            )
            cv2.line(
                right_lines,
                (0, y),
                (right.shape[1] - 1, y),
                (0, 255, 0),
                1,
            )
        cv2.imwrite(
            str(horizontal_path),
            np.hstack([left_lines, right_lines]),
        )

        print(f"Saved all matches: {all_matches_path}")
        print(f"Saved line pair: {horizontal_path}")

        self.finish()

    def finish(self):
        self.create_timer(0.2, self.stop)

    def stop(self):
        raise SystemExit


def main():
    rclpy.init()
    node = StereoGeometryDiagnostic()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
