# Waveshare IMX219-83 ISP profile

Platform:

- Jetson Orin Nano Super 8GB
- JetPack 6.2.2
- Jetson Linux R36.5
- Waveshare IMX219-83 Stereo Camera

Source:

https://www.waveshare.com/wiki/IMX219-83_Stereo_Camera

The original Waveshare profile removes the red side color cast, but contains
25 legacy attributes rejected by the R36.5 ISP parser.

`camera_overrides_jetpack_6_2_2.isp` keeps all accepted calibration data and
removes only attributes explicitly reported as invalid by nvargus-daemon.

System destination:

`/var/nvidia/nvcam/settings/camera_overrides.isp`
