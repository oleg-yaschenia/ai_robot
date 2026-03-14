#!/bin/bash
set -e

if [ -z "$1" ]; then
  echo "Usage: $0 /path/to/file.wav"
  exit 1
fi

echo "[robot_audio] Playing WAV file: $1"
aplay -D hw:1,0 "$1"
