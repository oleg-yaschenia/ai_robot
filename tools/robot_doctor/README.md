# robot-doctor v1.1

Read-only engineering diagnostic collector and deterministic health analyzer.

## Current capabilities

- collects Jetson/Linux, Git and ROS 2 diagnostics;
- stores snapshots outside the repository;
- treats the bounded `tegrastats` exit code 124 as expected;
- automatically creates:
  - `summary.txt`;
  - `snapshot.json`;
  - `health_report.md`;
  - `health_report.json`;
- checks:
  - expected ROS nodes;
  - Git working-tree cleanliness;
  - available memory;
  - disk usage;
  - failed systemd units;
  - `/dev/ttyTHS1`;
  - Jetson telemetry;
  - expected ROS package prefixes.

## Safety boundaries

It does not use sudo, modify parameters, restart services, command motors,
access firmware, or collect environment variables/API keys.

## Run

```bash
source /opt/ros/humble/setup.bash
source ~/ai_robot/ros2_ws/install/setup.bash

python3 ~/ai_robot/tools/robot_doctor/robot_doctor.py collect
```

View the latest report:

```bash
LATEST=$(
  python3 ~/ai_robot/tools/robot_doctor/robot_doctor.py latest
)

cat "$LATEST/summary.txt"
cat "$LATEST/health_report.md"
```

Reanalyze an existing snapshot:

```bash
python3 ~/ai_robot/tools/robot_doctor/robot_doctor.py \
  analyze \
  --snapshot "$LATEST"
```
