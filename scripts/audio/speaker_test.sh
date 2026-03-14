#!/bin/bash
set -e

echo "[robot_audio] Running stereo speaker test..."
speaker-test -D hw:1,0 -c 2 -r 48000 -F S32_LE -t sine

