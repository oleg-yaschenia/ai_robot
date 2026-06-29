#!/usr/bin/env python3
from pathlib import Path
import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


class StereoRectificationCheck(Node):
    def __init__(self):
        super().__init__("stereo_rectification_check")

        self.bridge = CvBridge()
        self.left_by_stamp = {}
        self.right_by_stamp = {}
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
            "Waiting for an exact left/right rectified pair..."
        )

    @staticmethod
    def stamp_key(msg: Image):
        return (
            int(msg.header.stamp.sec),
            int(msg.header.stamp.nanosec),
        )

    def left_cb(self, msg: Image):
        if self.done:
            return
        self.left_by_stamp[self.stamp_key(msg)] = msg
        self.try_pair()
        self.trim()

    def right_cb(self, msg: Image):
        if self.done:
            return
        self.right_by_stamp[self.stamp_key(msg)] = msg
        self.try_pair()
        self.trim()

    def trim(self):
        if len(self.left_by_stamp) > 30:
            for key in sorted(self.left_by_stamp)[:-20]:
                self.left_by_stamp.pop(key, None)
        if len(self.right_by_stamp) > 30:
            for key in sorted(self.right_by_stamp)[:-20]:
                self.right_by_stamp.pop(key, None)

    def try_pair(self):
        common = sorted(
            set(self.left_by_stamp).intersection(self.right_by_stamp)
        )
        if not common or self.done:
            return

        key = common[-1]
        left_msg = self.left_by_stamp[key]
        right_msg = self.right_by_stamp[key]
        self.done = True

        left = self.bridge.imgmsg_to_cv2(
            left_msg,
            desired_encoding="bgr8",
        )
        right = self.bridge.imgmsg_to_cv2(
            right_msg,
            desired_encoding="bgr8",
        )

        self.analyse_and_save(left, right, key)

    def analyse_and_save(self, left, right, stamp):
        gray_left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

        orb = cv2.ORB_create(nfeatures=2500)
        kp_left, des_left = orb.detectAndCompute(gray_left, None)
        kp_right, des_right = orb.detectAndCompute(gray_right, None)

        good = []
        if des_left is not None and des_right is not None:
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
            knn = matcher.knnMatch(des_left, des_right, k=2)
            for pair in knn:
                if len(pair) != 2:
                    continue
                m, n = pair
                if m.distance < 0.75 * n.distance:
                    good.append(m)

        dy_values = []
        disparities = []
        filtered = []

        for match in good:
            x_left, y_left = kp_left[match.queryIdx].pt
            x_right, y_right = kp_right[match.trainIdx].pt
            dy = abs(y_left - y_right)
            disparity = x_left - x_right

            if dy <= 25.0 and -16.0 <= disparity <= 512.0:
                dy_values.append(dy)
                disparities.append(disparity)
                filtered.append(match)

        print(f"Exact stamp: {stamp[0]}.{stamp[1]:09d}")
        print(f"Left keypoints: {len(kp_left)}")
        print(f"Right keypoints: {len(kp_right)}")
        print(f"Ratio-test matches: {len(good)}")
        print(f"Plausible matches: {len(filtered)}")

        if dy_values:
            dy_array = np.asarray(dy_values, dtype=np.float32)
            disp_array = np.asarray(disparities, dtype=np.float32)
            positive_ratio = float(np.mean(disp_array > 0.0))

            print(
                "Vertical error |dy|: "
                f"median={np.median(dy_array):.2f}px, "
                f"p90={np.percentile(dy_array, 90):.2f}px"
            )
            print(
                "Disparity: "
                f"median={np.median(disp_array):.2f}px, "
                f"positive_ratio={positive_ratio:.1%}"
            )
        else:
            print("No reliable feature matches for alignment estimate.")

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

        side_by_side = np.hstack([left_lines, right_lines])

        stamp_text = f"{stamp[0]}_{stamp[1]:09d}"
        pair_path = self.output_dir / f"rectified_pair_{stamp_text}.png"
        left_path = self.output_dir / f"left_rect_{stamp_text}.png"
        right_path = self.output_dir / f"right_rect_{stamp_text}.png"

        cv2.imwrite(str(pair_path), side_by_side)
        cv2.imwrite(str(left_path), left)
        cv2.imwrite(str(right_path), right)

        if filtered:
            selected = sorted(
                filtered,
                key=lambda item: item.distance,
            )[:100]
            matches_image = cv2.drawMatches(
                left,
                kp_left,
                right,
                kp_right,
                selected,
                None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
            )
            matches_path = (
                self.output_dir
                / f"rectified_matches_{stamp_text}.png"
            )
            cv2.imwrite(str(matches_path), matches_image)
            print(f"Saved matches: {matches_path}")

        print(f"Saved rectified pair: {pair_path}")
        print("Done.")

        self.create_timer(0.2, self.stop)

    def stop(self):
        raise SystemExit


def main():
    rclpy.init()
    node = StereoRectificationCheck()
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
