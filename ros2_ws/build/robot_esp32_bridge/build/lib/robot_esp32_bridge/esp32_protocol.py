import serial
import threading
import struct
import time


SOF1 = 0xAA
SOF2 = 0x55

# Head commands
CMD_SET_HEAD_MODE = 0x30
CMD_HEAD_MODE_ACK = 0x31
CMD_SET_HEAD_BRIGHTNESS = 0x32
CMD_SET_HEAD_LEVEL = 0x33
CMD_SET_HEAD_SPEAK_STYLE = 0x34

# Heartbeat
CMD_HEARTBEAT = 0x40

# Neck commands
CMD_SET_NECK_HOME = 0x50
CMD_SET_NECK_YAW = 0x51
CMD_SET_NECK_LIFT = 0x52
CMD_SET_NECK_PITCH = 0x53   # pitchOffset
CMD_SET_NECK_POSE = 0x54
CMD_NECK_ACK = 0x55


def crc8(data: bytes) -> int:
    crc = 0x00
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 0x80:
                crc = ((crc << 1) ^ 0x07) & 0xFF
            else:
                crc = (crc << 1) & 0xFF
    return crc


class Esp32Protocol:
    def __init__(self, port="/dev/ttyTHS1", baudrate=115200, timeout=0.05):
        self._ser = serial.Serial(port, baudrate, timeout=timeout)
        self._lock = threading.Lock()
        self._seq = 0

    def _next_seq(self) -> int:
        self._seq = (self._seq + 1) & 0xFF
        if self._seq == 0:
            self._seq = 1
        return self._seq

    def _build_packet(self, cmd: int, seq: int, payload: bytes = b"") -> bytes:
        # LEN = CMD + SEQ + PAYLOAD + CRC
        len_field = 3 + len(payload)
        body_wo_crc = bytes([len_field, cmd, seq]) + payload
        crc = crc8(body_wo_crc)
        return bytes([SOF1, SOF2]) + body_wo_crc + bytes([crc])

    def _read_packet_locked(self):
        b = self._ser.read(1)
        if not b:
            return None
        if b[0] != SOF1:
            return None

        b = self._ser.read(1)
        if not b or b[0] != SOF2:
            return None

        b = self._ser.read(1)
        if not b:
            return None

        len_field = b[0]
        content = self._ser.read(len_field)
        if len(content) != len_field:
            return None

        received_crc = content[-1]
        body_wo_crc = bytes([len_field]) + content[:-1]
        computed_crc = crc8(body_wo_crc)

        if received_crc != computed_crc:
            return None

        cmd = content[0]
        seq = content[1]
        payload = content[2:-1]

        return {
            "cmd": cmd,
            "seq": seq,
            "payload": payload,
        }

    def _send_packet(self, cmd: int, payload: bytes = b"", ack_cmd: int | None = None, ack_timeout: float = 1.0) -> bool:
        seq = self._next_seq()
        pkt = self._build_packet(cmd, seq, payload)

        with self._lock:
            self._ser.write(pkt)
            self._ser.flush()

            if ack_cmd is None:
                return True

            deadline = time.time() + ack_timeout
            while time.time() < deadline:
                rx = self._read_packet_locked()
                if rx is None:
                    continue

                # Ignore heartbeat and unrelated packets
                if rx["cmd"] == ack_cmd and rx["seq"] == seq:
                    return True

            return False

    # ---------------- Head ----------------

    def set_head_mode(self, mode: int) -> bool:
        return self._send_packet(
            CMD_SET_HEAD_MODE,
            bytes([mode & 0xFF]),
            ack_cmd=CMD_HEAD_MODE_ACK,
        )

    def set_head_brightness(self, brightness: int) -> bool:
        return self._send_packet(
            CMD_SET_HEAD_BRIGHTNESS,
            bytes([brightness & 0xFF]),
            ack_cmd=CMD_HEAD_MODE_ACK,
        )

    def set_head_level(self, level: int) -> bool:
        return self._send_packet(
            CMD_SET_HEAD_LEVEL,
            bytes([level & 0xFF]),
            ack_cmd=CMD_HEAD_MODE_ACK,
        )

    def set_head_speaking_style(self, style: int) -> bool:
        return self._send_packet(
            CMD_SET_HEAD_SPEAK_STYLE,
            bytes([style & 0xFF]),
            ack_cmd=CMD_HEAD_MODE_ACK,
        )

    # ---------------- Neck ----------------

    def neck_home(self) -> bool:
        return self._send_packet(
            CMD_SET_NECK_HOME,
            b"",
            ack_cmd=CMD_NECK_ACK,
        )

    def neck_set_yaw(self, yaw: int) -> bool:
        payload = struct.pack("<h", int(yaw))
        return self._send_packet(
            CMD_SET_NECK_YAW,
            payload,
            ack_cmd=CMD_NECK_ACK,
        )

    def neck_set_lift(self, lift: int) -> bool:
        payload = struct.pack("<h", int(lift))
        return self._send_packet(
            CMD_SET_NECK_LIFT,
            payload,
            ack_cmd=CMD_NECK_ACK,
        )

    def neck_set_pitch_offset(self, pitch_offset: int) -> bool:
        payload = struct.pack("<h", int(pitch_offset))
        return self._send_packet(
            CMD_SET_NECK_PITCH,
            payload,
            ack_cmd=CMD_NECK_ACK,
        )

    def neck_set_pose(self, yaw: int, lift: int, pitch_offset: int) -> bool:
        payload = struct.pack("<hhh", int(yaw), int(lift), int(pitch_offset))
        return self._send_packet(
            CMD_SET_NECK_POSE,
            payload,
            ack_cmd=CMD_NECK_ACK,
        )

    def close(self) -> None:
        with self._lock:
            if self._ser and self._ser.is_open:
                self._ser.close()
