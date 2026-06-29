# Baseline Guardian v1

Creates a version-controlled baseline from a healthy robot-doctor snapshot
and compares later snapshots against it.

## Protected areas

- required ROS nodes;
- SHA-256 hashes of key perception and stereo files;
- selected Python package versions;
- ROS package prefixes;
- available memory relative to the baseline;
- root and home disk usage relative to the baseline.

## Create the perception-v5 baseline

```bash
LATEST=$(
  python3 ~/ai_robot/tools/robot_doctor/robot_doctor.py latest
)

python3 \
  ~/ai_robot/tools/robot_doctor/baseline_guardian.py \
  create \
  --snapshot "$LATEST" \
  --name stable-perception-v5 \
  --output \
  ~/ai_robot/config/robot_doctor/baselines/perception_v5.json
```

The command refuses to overwrite an existing baseline unless `--force`
is explicitly supplied.

## Compare the current snapshot

```bash
python3 \
  ~/ai_robot/tools/robot_doctor/baseline_guardian.py \
  compare \
  --snapshot "$LATEST" \
  --baseline \
  ~/ai_robot/config/robot_doctor/baselines/perception_v5.json
```

Reports are written outside Git:

```text
~/ai_robot_artifacts/robot_doctor/comparisons/
```

A non-OK comparison returns exit code 2 so it can later be used in CI.
