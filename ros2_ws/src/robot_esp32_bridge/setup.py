from glob import glob
import os

from setuptools import setup

package_name = "robot_esp32_bridge"

setup(
    name=package_name,
    version="0.0.2",
    packages=[package_name],
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            ["resource/" + package_name],
        ),
        ("share/" + package_name, ["package.xml"]),
        (
            os.path.join("share", package_name, "launch"),
            glob("launch/*.launch.py"),
        ),
    ],
    install_requires=["setuptools", "pyserial"],
    zip_safe=True,
    maintainer="warxen",
    maintainer_email="yaschenia@gmail.com",
    description="Single-owner ROS 2 transport for Jetson <-> ESP32",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "esp32_bridge_node = robot_esp32_bridge.esp32_bridge_node:main",
            "head_demo_publisher = robot_esp32_bridge.head_demo_publisher:main",
            "head_state_manager = robot_esp32_bridge.head_state_manager:main",
            "neck_demo_publisher = robot_esp32_bridge.neck_demo_publisher:main",
            "neck_state_manager = robot_esp32_bridge.neck_state_manager:main",
            # Retained only to print a clear migration error. They no longer
            # open UART and therefore cannot conflict with esp32_bridge_node.
            "drive_bridge_node = robot_esp32_bridge.drive_bridge_node:main",
            "neck_bridge_node = robot_esp32_bridge.neck_bridge_node:main",
        ],
    },
)
