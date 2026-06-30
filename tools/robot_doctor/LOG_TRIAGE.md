# Log Triage v1

Read-only deterministic analyzer for recent robot engineering and ROS logs.

## What it does

- discovers command logs from the latest Regression Agent and Robot Doctor runs;
- scans recent ROS build/test/runtime logs when present;
- classifies errors and warnings;
- groups repeated messages after removing timestamps and volatile numbers;
- separates configured benign warnings;
- identifies a primary issue;
- proposes only safe diagnostic checks;
- writes Markdown and JSON reports outside Git.

## Safety

It does not:

- use `sudo`;
- restart services;
- change ROS parameters;
- flash firmware;
- send motor, servo or UART commands;
- edit source code or configuration.

## Run

```bash
python3 \
  ~/ai_robot/tools/robot_doctor/log_triage.py
```

View the report:

```bash
TRIAGE_DIR=$(
  cat ~/ai_robot_artifacts/robot_doctor/log_triage/latest.txt
)

cat "$TRIAGE_DIR/log_triage_report.md"
```

Scan an additional log file or directory:

```bash
python3 \
  ~/ai_robot/tools/robot_doctor/log_triage.py \
  --path /path/to/log_or_directory
```

Exit codes:

- `0`: no active issues;
- `2`: active warnings;
- `3`: active errors;
- `1`: tool execution failure.
