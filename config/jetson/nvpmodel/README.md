# Jetson Orin Nano Super power profile

## Profile

    25W_BALANCE

## Limits

- CPU cores: 6
- CPU maximum: 1497600 kHz
- GPU maximum: 816000000 Hz
- EMC settings inherited from NVIDIA 25W profile
- Dynamic CPU and GPU frequency scaling remains enabled
- OC1, OC2 and OC3 protections remain enabled

## Purpose

The NVIDIA 25W profile limits the CPU to 1344000 kHz. The robot workload
contains both GPU-heavy inference and CPU-heavy stereo processing.

The custom profile raises the CPU maximum to the frequency used by the NVIDIA
15W mode while retaining the faster memory configuration of the 25W mode.
The GPU maximum is limited to 816 MHz to reduce instantaneous power peaks.

## Validated workload

- Jetson Orin Nano Super 8GB
- JetPack 6.2.2
- ROS2 Humble
- Stereo IMX219 cameras at 1024x768
- image_proc rectification
- StereoBM disparity
- YOLO perception
- local assistant contour
- Qwen runtime under additional load

No OC3 events were observed during the validation load.

## Normal operation

Do not run:

    sudo jetson_clocks

The profile is intended to use dynamic frequency scaling.

## Installation

Run:

    sudo python3 \
      tools/jetson/install_nvpmodel_cpu_balanced.py

Then select the profile using its generated mode ID and reboot.

## Verification

Run:

    ./tools/jetson/verify_25w_balance.sh

The profile must be reinstalled and revalidated after JetPack, BSP or
nvpmodel package updates.
