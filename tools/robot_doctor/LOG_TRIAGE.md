# Log Triage v1.1

The first version used broad keyword matching and produced false positives
from successful colcon debug lines, package license text, ttyTHS1 listings,
and ROS doctor headings.

v1.1 scans only relevant files, requires explicit error/warning markers,
categorizes only confirmed issues, and includes a built-in self-test.

## Self-test

```bash
python3 ~/ai_robot/tools/robot_doctor/log_triage.py --self-test
```

Expected: `Self-test: 10/10 passed`

## Run

```bash
python3 ~/ai_robot/tools/robot_doctor/log_triage.py
echo "EXIT_CODE=$?"
```

Reports remain outside Git under:

```text
~/ai_robot_artifacts/robot_doctor/log_triage/
```
