# Configuration Review Agent v1.1

Read-only deterministic review of ROS 2 launch and configuration files.

## Checks

- Python syntax for `*.launch.py`;
- YAML, JSON and XML syntax;
- duplicate YAML keys;
- unresolved Git conflict markers;
- dangerous host/shell/firmware operations;
- hard-coded `/home/<user>/...` paths;
- missing directly referenced local model/config files, including `PathJoinSubstitution` handling;
- duplicate topic remap sources inside literal `remappings` lists;
- selected unsafe boolean settings.

## Safety

The tool never launches ROS, changes parameters, writes UART, flashes ESP32,
runs sudo, or edits any project file.

Reports are written outside Git:

```text
~/ai_robot_artifacts/robot_doctor/configuration_reviews/<timestamp>/
```

## Self-test

```bash
python3 \
  ~/ai_robot/tools/robot_doctor/config_review.py \
  --self-test
```

Expected:

```text
Self-test: 5/5 passed
```

## Run

```bash
python3 \
  ~/ai_robot/tools/robot_doctor/config_review.py

echo "EXIT_CODE=$?"
```

Exit codes:

- `0`: OK;
- `2`: warnings;
- `3`: errors;
- `1`: tool failure.

## v1.1

- excludes historical baseline snapshots from portability checks;
- resolves literal suffixes inside `PathJoinSubstitution`;
- checks relative model references under both repository root and `ros2_ws`.
