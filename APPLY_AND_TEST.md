# ESP32 conflict fix: apply and test

These files convert the Jetson <-> ESP32 connection to a single-owner UART
architecture. Only `esp32_bridge_node` opens `/dev/ttyTHS1`.

## 1. Copy the replacement files

Copy this directory over the repository root while preserving paths.

## 2. Remove generated files from Git tracking

The commands below do not delete local runtime files when `--cached` is used.

```bash
cd ~/ai_robot

git rm -r --cached ros2_ws/build ros2_ws/install ros2_ws/log
git rm --cached ros2_ws/stereo.wav ros2_ws/yolo11s.pt
```

The model and WAV stay on the Jetson working tree and are ignored by Git.

## 3. Set the repository root once

```bash
export AI_ROBOT_ROOT="$HOME/ai_robot"
```

Add the same line to `~/.bashrc` after verification.

## 4. Clean only the affected installed packages and rebuild

```bash
cd ~/ai_robot/ros2_ws
rm -rf \
  build/robot_esp32_bridge install/robot_esp32_bridge \
  build/robot_bringup install/robot_bringup \
  build/robot_vision_assistant install/robot_vision_assistant

source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select \
  robot_esp32_bridge robot_bringup robot_vision_assistant
source install/setup.bash
```

## 5. Check the unified bridge alone

```bash
ros2 launch robot_esp32_bridge esp32_bridge.launch.py
```

In another terminal:

```bash
source /opt/ros/humble/setup.bash
source ~/ai_robot/ros2_ws/install/setup.bash

ros2 node list | sort
sudo lsof /dev/ttyTHS1
```

Expected:

- exactly one `esp32_bridge_node`;
- exactly one process holding `/dev/ttyTHS1`;
- no running `drive_bridge_node` or `neck_bridge_node`.

## 6. Test movement and watchdog

Raise the wheels before this test.

```bash
ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist \
"{linear: {x: 0.05, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}"
```

The bridge must stop the wheels once after `cmd_timeout_sec`; it must not flood
UART with repeated STOP packets.

## 7. Test the head without the assistant

```bash
ros2 launch robot_esp32_bridge head_standalone.launch.py
```

Do not run this together with `robot_base.launch.py`, because both launch files
intentionally start the single transport for their standalone scenarios.

## 8. Test the complete robot

```bash
ros2 launch robot_bringup robot_assistant_full.launch.py
```

Then verify:

```bash
ros2 node list | grep -E 'esp32|drive|neck|head'
sudo lsof /dev/ttyTHS1
ros2 topic pub --once /voice/start std_msgs/msg/String "{data: 'start'}"
```

Expected UART owner: only `esp32_bridge_node`.

## 9. Commit

```bash
cd ~/ai_robot
git add .
git commit -m "Unify ESP32 UART ownership and clean launch configuration"
git push
```
