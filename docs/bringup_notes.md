# Bringup notes

## Stereo camera
- Working command:
  env -u DISPLAY -u WAYLAND_DISPLAY -u XAUTHORITY python3 stereo_cam.py
- Script path:
  /home/warxen/ai_robot/scripts/camera/stereo_cam.py
- Wrapper script:
  /home/warxen/ai_robot/scripts/camera/stereo_capture_test.sh
- Notes:
  Headless stereo capture via Argus.
  Press Enter to save left/right image pair.
  Type q then Enter to quit.
- Expected output:
  Saved stereo pairs into stereo_pairs/
- Device/sensor-id:
  left = sensor-id 0
  right = sensor-id 1
- Stream mode:
  1920x1080 @ 30
- Fullres mode:
  3280x2464 @ 21

## Stereo microphones
- Working command:
  arecord -D hw:1,0 -c 2 -r 48000 -f S32_LE -d 5 stereo.wav
- ALSA device:
  hw:1,0
- Sample rate:
  48000
- Format:
  S32_LE
- Channels:
  2
- Notes:
  Stereo capture works correctly.

## Stereo speakers
- Left/right test command:
  speaker-test -D hw:1,0 -c 2 -r 48000 -F S32_LE -t sine
- WAV playback command:
  aplay -D hw:1,0 song.wav
- Output device:
  hw:1,0
- Sample rate:
  48000
- Format:
  S32_LE
- Channels:
  2
- Notes:
  Stereo speaker output works correctly.
