#!/usr/bin/env bash

set -euo pipefail

EXPECTED_MODE="25W_BALANCE"
EXPECTED_CPU_MAX="1497600"
EXPECTED_GPU_MAX="816000000"

ERRORS=0

MODE_OUTPUT="$(sudo nvpmodel -q 2>&1)"

echo "=== Power mode ==="
echo "$MODE_OUTPUT"

if ! grep -q "NV Power Mode: ${EXPECTED_MODE}" <<< "$MODE_OUTPUT"; then
    echo "ERROR: active mode is not ${EXPECTED_MODE}"
    ERRORS=$((ERRORS + 1))
fi

echo
echo "=== CPU limits ==="

for CPU in 0 1 2 3 4 5; do
    PATH_MAX="/sys/devices/system/cpu/cpu${CPU}/cpufreq/scaling_max_freq"
    VALUE="$(cat "$PATH_MAX")"

    echo "cpu${CPU}: ${VALUE} kHz"

    if [ "$VALUE" != "$EXPECTED_CPU_MAX" ]; then
        echo "ERROR: cpu${CPU} expected ${EXPECTED_CPU_MAX}"
        ERRORS=$((ERRORS + 1))
    fi
done

GPU_PATH="/sys/devices/platform/17000000.gpu/devfreq_dev/max_freq"
GPU_VALUE="$(cat "$GPU_PATH")"

echo
echo "=== GPU limit ==="
echo "GPU: ${GPU_VALUE} Hz"

if [ "$GPU_VALUE" != "$EXPECTED_GPU_MAX" ]; then
    echo "ERROR: GPU expected ${EXPECTED_GPU_MAX}"
    ERRORS=$((ERRORS + 1))
fi

echo
echo "=== OC counters ==="

for DIR in /sys/class/hwmon/hwmon*; do
    if [ "$(cat "$DIR/name" 2>/dev/null || true)" = "soctherm_oc" ]; then
        grep -H . "$DIR"/oc*_event_cnt 2>/dev/null || true
    fi
done

echo
if [ "$ERRORS" -eq 0 ]; then
    echo "25W_BALANCE verification passed"
    exit 0
fi

echo "Verification failed with ${ERRORS} error(s)"
exit 1
