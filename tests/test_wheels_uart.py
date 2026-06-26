#!/usr/bin/env python3
import argparse
import struct
import time
import serial

SOF1 = 0xAA
SOF2 = 0x55

CMD_PING = 0x10
CMD_PONG = 0x11

CMD_SET_WHEEL_SPEED = 0x70
CMD_SET_ALL_WHEEL_SPEEDS = 0x71
CMD_WHEELS_STOP = 0x72
CMD_WHEELS_ACK = 0x73

CMD_SET_CHASSIS = 0x74
CMD_CHASSIS_ACK = 0x75

CMD_GET_YAW_TELEMETRY = 0x76
CMD_YAW_TELEMETRY = 0x77
CMD_GET_WHEEL_SPEEDS_1 = 0x78
CMD_WHEEL_SPEEDS_1 = 0x79
CMD_GET_WHEEL_SPEEDS_2 = 0x7A
CMD_WHEEL_SPEEDS_2 = 0x7B

WHEEL_IDS = {
    "fl": 0,
    "rl": 1,
    "fr": 2,
    "rr": 3,
}


def crc8(data: bytes) -> int:
    crc = 0
    for b in data:
        crc ^= b
        for _ in range(8):
            if crc & 0x80:
                crc = ((crc << 1) ^ 0x07) & 0xFF
            else:
                crc = (crc << 1) & 0xFF
    return crc


class Esp32WheelClient:
    def __init__(self, port: str = "/dev/ttyTHS1", baud: int = 115200, timeout: float = 0.1):
        self.ser = serial.Serial(port, baudrate=baud, timeout=timeout)
        self.seq = 0

    def close(self):
        if self.ser and self.ser.is_open:
            self.ser.close()

    def next_seq(self) -> int:
        s = self.seq
        self.seq = (self.seq + 1) & 0xFF
        return s

    def build_packet(self, cmd: int, payload: bytes = b"", seq: int | None = None) -> bytes:
        if seq is None:
            seq = self.next_seq()

        length = 3 + len(payload)  # CMD + SEQ + PAYLOAD + CRC
        core = bytes([length, cmd, seq]) + payload
        crc = crc8(core)
        return bytes([SOF1, SOF2]) + core + bytes([crc])

    def send_packet(self, cmd: int, payload: bytes = b"", seq: int | None = None) -> int:
        if seq is None:
            seq = self.next_seq()
        pkt = self.build_packet(cmd, payload, seq)
        self.ser.write(pkt)
        self.ser.flush()
        return seq

    def read_packet(self, timeout: float = 1.0):
        deadline = time.time() + timeout

        def read_exact(n: int):
            data = bytearray()
            while len(data) < n and time.time() < deadline:
                chunk = self.ser.read(n - len(data))
                if chunk:
                    data.extend(chunk)
            return bytes(data)

        while time.time() < deadline:
            b = self.ser.read(1)
            if not b:
                continue

            if b[0] != SOF1:
                continue

            b2 = self.ser.read(1)
            if not b2 or b2[0] != SOF2:
                continue

            hdr = self.ser.read(1)
            if not hdr:
                continue

            length = hdr[0]
            rest = read_exact(length)
            if len(rest) != length:
                continue

            cmd = rest[0]
            seq = rest[1]
            payload = rest[2:-1]
            rx_crc = rest[-1]

            calc_crc = crc8(bytes([length]) + rest[:-1])
            if calc_crc != rx_crc:
                continue

            return {
                "cmd": cmd,
                "seq": seq,
                "payload": payload,
            }

        return None

    def drain_input(self, duration: float = 0.05):
        deadline = time.time() + duration
        while time.time() < deadline:
            pkt = self.read_packet(timeout=0.01)
            if pkt is None:
                continue

    def ping(self) -> bool:
        seq = self.send_packet(CMD_PING, b"")
        deadline = time.time() + 1.0

        while time.time() < deadline:
            pkt = self.read_packet(timeout=0.1)
            if pkt is None:
                continue
            if pkt["cmd"] == 0x40:
                continue
            return pkt["cmd"] == CMD_PONG and pkt["seq"] == seq
        return False

    def set_wheel_speed(self, wheel: str, speed: float) -> bool:
        wheel = wheel.lower()
        if wheel not in WHEEL_IDS:
            raise ValueError(f"Unknown wheel: {wheel}")

        payload = bytes([WHEEL_IDS[wheel]]) + struct.pack("<f", speed)
        seq = self.send_packet(CMD_SET_WHEEL_SPEED, payload)

        deadline = time.time() + 1.0
        while time.time() < deadline:
            pkt = self.read_packet(timeout=0.1)
            if pkt is None:
                continue
            if pkt["cmd"] == 0x40:
                continue
            return pkt["cmd"] == CMD_WHEELS_ACK and pkt["seq"] == seq
        return False

    def set_all_wheel_speeds(self, fl: float, rl: float, fr: float, rr: float) -> bool:
        payload = struct.pack("<ffff", fl, rl, fr, rr)
        seq = self.send_packet(CMD_SET_ALL_WHEEL_SPEEDS, payload)

        deadline = time.time() + 1.0
        while time.time() < deadline:
            pkt = self.read_packet(timeout=0.1)
            if pkt is None:
                continue
            if pkt["cmd"] == 0x40:
                continue
            return pkt["cmd"] == CMD_WHEELS_ACK and pkt["seq"] == seq
        return False

    def set_chassis(self, vx: float, vy: float, wz: float) -> bool:
        payload = struct.pack("<fff", vx, vy, wz)
        seq = self.send_packet(CMD_SET_CHASSIS, payload)

        deadline = time.time() + 1.0
        while time.time() < deadline:
            pkt = self.read_packet(timeout=0.1)
            if pkt is None:
                continue
            if pkt["cmd"] == 0x40:
                continue
            return pkt["cmd"] == CMD_CHASSIS_ACK and pkt["seq"] == seq
        return False

    def stop_all(self) -> bool:
        seq = self.send_packet(CMD_WHEELS_STOP, b"")

        deadline = time.time() + 1.0
        while time.time() < deadline:
            pkt = self.read_packet(timeout=0.1)
            if pkt is None:
                continue
            if pkt["cmd"] == 0x40:
                continue
            return pkt["cmd"] == CMD_WHEELS_ACK and pkt["seq"] == seq
        return False

    def get_yaw_telemetry(self):
        self.drain_input(0.03)
        seq = self.send_packet(CMD_GET_YAW_TELEMETRY, b"")
        deadline = time.time() + 0.8

        while time.time() < deadline:
            pkt = self.read_packet(timeout=0.05)
            if pkt is None:
                continue
            if pkt["cmd"] == 0x40:
                continue
            if pkt["cmd"] == CMD_YAW_TELEMETRY and pkt["seq"] == seq:
                if len(pkt["payload"]) != 4:
                    return None
                (yaw,) = struct.unpack("<f", pkt["payload"])
                return yaw
        return None

    def get_wheel_speeds_1(self):
        self.drain_input(0.03)
        seq = self.send_packet(CMD_GET_WHEEL_SPEEDS_1, b"")
        deadline = time.time() + 0.8

        while time.time() < deadline:
            pkt = self.read_packet(timeout=0.05)
            if pkt is None:
                continue
            if pkt["cmd"] == 0x40:
                continue
            if pkt["cmd"] == CMD_WHEEL_SPEEDS_1 and pkt["seq"] == seq:
                if len(pkt["payload"]) != 8:
                    return None
                fl, rl = struct.unpack("<ff", pkt["payload"])
                return {"fl": fl, "rl": rl}
        return None

    def get_wheel_speeds_2(self):
        self.drain_input(0.03)
        seq = self.send_packet(CMD_GET_WHEEL_SPEEDS_2, b"")
        deadline = time.time() + 0.8

        while time.time() < deadline:
            pkt = self.read_packet(timeout=0.05)
            if pkt is None:
                continue
            if pkt["cmd"] == 0x40:
                continue
            if pkt["cmd"] == CMD_WHEEL_SPEEDS_2 and pkt["seq"] == seq:
                if len(pkt["payload"]) != 8:
                    return None
                fr, rr = struct.unpack("<ff", pkt["payload"])
                return {"fr": fr, "rr": rr}
        return None


def cmd_ping(client: Esp32WheelClient) -> int:
    ok = client.ping()
    print("PING:", ok)
    return 0 if ok else 1


def cmd_one(client: Esp32WheelClient, wheel: str, speed: float, run_s: float, pause_s: float) -> int:
    print(f"TEST {wheel.upper()}: {speed:+.1f}")
    ok = client.set_wheel_speed(wheel, speed)
    print("  ACK:", ok)
    if not ok:
        return 1

    time.sleep(run_s)

    ok = client.stop_all()
    print("  STOP ACK:", ok)
    if not ok:
        return 1

    time.sleep(pause_s)
    return 0


def cmd_test_all(client: Esp32WheelClient, speed: float, run_s: float, pause_s: float) -> int:
    for wheel in ("fl", "rl", "fr", "rr"):
        rc = cmd_one(client, wheel, speed, run_s, pause_s)
        if rc != 0:
            return rc
    return 0


def cmd_test_all4(client: Esp32WheelClient, speed: float, run_s: float, pause_s: float) -> int:
    print(f"TEST ALL via 4x CMD_SET_WHEEL_SPEED: FL/RL/FR/RR = {speed:+.1f}")

    for wheel in ("fl", "rl", "fr", "rr"):
        ok = client.set_wheel_speed(wheel, speed)
        print(f"  {wheel.upper()} ACK:", ok)
        if not ok:
            return 1
        time.sleep(0.05)

    time.sleep(run_s)

    ok = client.stop_all()
    print("  STOP ACK:", ok)
    if not ok:
        return 1

    time.sleep(pause_s)
    return 0


def cmd_stop(client: Esp32WheelClient) -> int:
    ok = client.stop_all()
    print("STOP ACK:", ok)
    return 0 if ok else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="/dev/ttyTHS1")
    parser.add_argument("--baud", type=int, default=115200)

    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("ping")

    p_one = sub.add_parser("one")
    p_one.add_argument("wheel", choices=["fl", "rl", "fr", "rr"])
    p_one.add_argument("speed", type=float)
    p_one.add_argument("--run-s", type=float, default=1.5)
    p_one.add_argument("--pause-s", type=float, default=0.8)

    p_all = sub.add_parser("test_all")
    p_all.add_argument("--speed", type=float, default=800.0)
    p_all.add_argument("--run-s", type=float, default=1.5)
    p_all.add_argument("--pause-s", type=float, default=0.8)

    p_all4 = sub.add_parser("test_all4")
    p_all4.add_argument("--speed", type=float, default=800.0)
    p_all4.add_argument("--run-s", type=float, default=1.5)
    p_all4.add_argument("--pause-s", type=float, default=0.8)

    sub.add_parser("stop")

    args = parser.parse_args()
    client = Esp32WheelClient(port=args.port, baud=args.baud)

    try:
        if args.cmd == "ping":
            raise SystemExit(cmd_ping(client))
        elif args.cmd == "one":
            raise SystemExit(cmd_one(client, args.wheel, args.speed, args.run_s, args.pause_s))
        elif args.cmd == "test_all":
            raise SystemExit(cmd_test_all(client, args.speed, args.run_s, args.pause_s))
        elif args.cmd == "test_all4":
            raise SystemExit(cmd_test_all4(client, args.speed, args.run_s, args.pause_s))
        elif args.cmd == "stop":
            raise SystemExit(cmd_stop(client))
    finally:
        client.close()


if __name__ == "__main__":
    main()
