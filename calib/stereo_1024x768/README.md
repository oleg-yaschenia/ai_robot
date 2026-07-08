# Stereo calibration 1024x768

Hardware:

- Waveshare IMX219-83 Stereo Camera
- Jetson Orin Nano Super 8GB
- JetPack 6.2.2 / Jetson Linux R36.5

Capture pipeline:

- Sensor mode: 1640x1232
- ROS output: 1024x768
- Frame rate: 10 FPS
- Left sensor ID: 1
- Right sensor ID: 0
- Hardware downscaling: nvvidconv

Calibration target:

- Chessboard cells: 9x7
- Internal corners: 8x6
- Square size: 27.7 mm

Calibration result:

- Stereo baseline: 0.059291 m
- Rectification median vertical error: 0.763 px
- Rectification mean vertical error: 0.737 px
- Rectification maximum vertical error: 1.250 px
- Status: accepted

Disparity baseline:

- Algorithm: StereoBM
- Minimum disparity: 0
- Disparity range: 128
- Correlation window: 15
- Measured output rate: approximately 7.8 Hz
- Minimum measurable distance: approximately 0.344 m
