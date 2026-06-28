# Vision Evaluation Pack v1

This pack records the robot's stereo images together with the current YOLO
perception JSON and interpreted-scene JSON. It provides an objective baseline
before replacing or extending the detector.

## Install in the repository

Copy `tools/vision_eval` to `~/ai_robot/tools/vision_eval` and make scripts executable:

```bash
chmod +x ~/ai_robot/tools/vision_eval/*.py
```

No separate build is required. Use a terminal where ROS 2 and the workspace are sourced.

## Verify topics

```bash
ros2 topic hz /camera/left/image_raw
ros2 topic hz /camera/right/image_raw
ros2 topic echo /perception/state_json --once
ros2 topic echo /scene/interpreted_json --once
```

## Initial three-scenario smoke dataset

Run the full robot launch in terminal 1. In terminal 2:

```bash
source /opt/ros/humble/setup.bash
source ~/ai_robot/ros2_ws/install/setup.bash
export AI_ROBOT_ROOT="$HOME/ai_robot"
```

Empty scene, nobody in front of the robot:

```bash
python3 ~/ai_robot/tools/vision_eval/capture_ros_samples.py \
  --scenario empty_room \
  --people 0 \
  --objects "chair" \
  --lighting normal \
  --distance medium \
  --count 5 \
  --notes "Test room without a person"
```

One person in front of the robot:

```bash
python3 ~/ai_robot/tools/vision_eval/capture_ros_samples.py \
  --scenario person_front_near \
  --people 1 \
  --objects "chair" \
  --lighting normal \
  --distance near \
  --count 5 \
  --notes "One person facing the robot"
```

One person showing a cup:

```bash
python3 ~/ai_robot/tools/vision_eval/capture_ros_samples.py \
  --scenario object_in_hand_cup \
  --people 1 \
  --objects "cup" \
  --lighting normal \
  --distance near \
  --count 5 \
  --notes "Cup clearly held in hand"
```

Use only labels that are truly present in the frame. The current YOLO node is
restricted to person, cat, dog, cup, bottle, cell phone, laptop and chair, but
future evaluators may use any object labels.

## Analyze the current baseline

```bash
python3 ~/ai_robot/tools/vision_eval/analyze_presence.py
```

Outputs:

- `~/ai_robot/data/vision_eval/manifest.jsonl`
- stereo PNG files under `images/left` and `images/right`
- `~/ai_robot/data/vision_eval/baseline_presence.csv`

## Important

This v1 report measures class presence and exact person count. It does not yet
measure bounding-box mAP, tracking stability, open-vocabulary recognition or VLM
answer quality. Those are added after the capture pipeline is confirmed.
