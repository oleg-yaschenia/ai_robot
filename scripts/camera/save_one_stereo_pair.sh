#!/bin/bash
set -e

cd /home/warxen/ai_robot/scripts/camera
env -u DISPLAY -u WAYLAND_DISPLAY -u XAUTHORITY python3 /home/warxen/ai_robot/scripts/camera/save_one_stereo_pair.py
