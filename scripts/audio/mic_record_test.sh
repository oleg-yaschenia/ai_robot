#!/bin/bash
set -e

OUT_DIR="/home/warxen/ai_robot/data/recordings"
mkdir -p "$OUT_DIR"

echo "[robot_audio] Recording 5 seconds from stereo microphones..."
arecord -D hw:1,0 -c 2 -r 48000 -f S32_LE -d 5 "$OUT_DIR/stereo_test.wav"

echo "[robot_audio] Saved to $OUT_DIR/stereo_test.wav"
