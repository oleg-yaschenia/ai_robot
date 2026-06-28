#!/usr/bin/env python3

"""Deprecated compatibility entry point.

UART ownership was moved to esp32_bridge_node. This module deliberately does
not open the serial port, preventing accidental multiple owners of ttyTHS1.
"""


def main(args=None) -> None:
    del args
    raise SystemExit(
        "neck_bridge_node is retired. Start esp32_bridge_node or use "
        "robot_esp32_bridge/esp32_bridge.launch.py."
    )


if __name__ == "__main__":
    main()
