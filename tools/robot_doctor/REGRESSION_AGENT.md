# Regression Agent v1

Deterministic verification pipeline for the AI robot project.

## Pipeline

1. `git diff --check HEAD`
2. Python syntax compilation for tracked project files
3. `colcon build` for selected ROS 2 packages
4. fresh `robot-doctor` snapshot
5. Baseline Guardian comparison
6. final `PASS`, `WARNING`, or `FAIL` report

## Safety

The agent never:

- runs `sudo`;
- flashes ESP32;
- starts or restarts robot services;
- changes ROS parameters;
- sends motor, servo, UART, or actuator commands.

The full robot stack must already be running for health and baseline checks.

## Run

```bash
source /opt/ros/humble/setup.bash
source ~/ai_robot/ros2_ws/install/setup.bash

python3 \
  ~/ai_robot/tools/robot_doctor/regression_agent.py
```

Reports are stored outside Git:

```text
~/ai_robot_artifacts/robot_doctor/regressions/<timestamp>/
```

View the latest report:

```bash
RUN_DIR=$(
  cat ~/ai_robot_artifacts/robot_doctor/regressions/latest.txt
)

cat "$RUN_DIR/regression_report.md"
```

## Optional modes

Syntax/build only:

```bash
python3 \
  ~/ai_robot/tools/robot_doctor/regression_agent.py \
  --skip-health \
  --skip-baseline
```

Skip build during rapid iteration:

```bash
python3 \
  ~/ai_robot/tools/robot_doctor/regression_agent.py \
  --skip-build
```

Skipped checks produce `WARNING`, never `PASS`.
