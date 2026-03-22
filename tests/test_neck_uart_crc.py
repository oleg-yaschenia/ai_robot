import serial
import struct
import time

PORT = "/dev/ttyTHS1"
BAUD = 115200

# Подставь реальные значения из config.h
PROTO_SOF1 = 0xAA
PROTO_SOF2 = 0x55

CMD_SET_NECK_HOME  = 0x50
CMD_SET_NECK_YAW   = 0x51
CMD_SET_NECK_LIFT  = 0x52
CMD_SET_NECK_PITCH = 0x53   # pitchOffset
CMD_SET_NECK_POSE  = 0x54
CMD_NECK_ACK       = 0x55


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


class Esp32NeckClient:
    def __init__(self, port: str, baud: int):
        self.ser = serial.Serial(port, baudrate=baud, timeout=0.2)
        self.seq = 1

    def next_seq(self) -> int:
        s = self.seq
        self.seq = (self.seq + 1) & 0xFF
        if self.seq == 0:
            self.seq = 1
        return s

    def build_packet(self, cmd: int, seq: int, payload: bytes = b"") -> bytes:
        # LEN = CMD + SEQ + PAYLOAD + CRC
        len_field = 3 + len(payload)

        body_wo_crc = bytes([len_field, cmd, seq]) + payload
        crc = crc8(body_wo_crc)

        return bytes([PROTO_SOF1, PROTO_SOF2]) + body_wo_crc + bytes([crc])

    def send_packet(self, cmd: int, payload: bytes = b"") -> int:
        seq = self.next_seq()
        pkt = self.build_packet(cmd, seq, payload)
        self.ser.write(pkt)
        self.ser.flush()
        print("TX:", pkt.hex(" "))
        return seq

    def read_packet(self):
        b = self.ser.read(1)
        if not b:
            return None
        if b[0] != PROTO_SOF1:
            return None

        b = self.ser.read(1)
        if not b or b[0] != PROTO_SOF2:
            return None

        b = self.ser.read(1)
        if not b:
            return None

        len_field = b[0]
        content = self.ser.read(len_field)
        if len(content) != len_field:
            return None

        received_crc = content[-1]
        body_wo_crc = bytes([len_field]) + content[:-1]
        computed_crc = crc8(body_wo_crc)

        if received_crc != computed_crc:
            print(
                f"RX CRC ERROR: recv=0x{received_crc:02X} calc=0x{computed_crc:02X} "
                f"raw={(bytes([PROTO_SOF1, PROTO_SOF2, len_field]) + content).hex(' ')}"
            )
            return None

        cmd = content[0]
        seq = content[1]
        payload = content[2:-1]

        pkt = {
            "cmd": cmd,
            "seq": seq,
            "payload": payload,
        }
        print("RX:", (bytes([PROTO_SOF1, PROTO_SOF2, len_field]) + content).hex(" "))
        return pkt

    def wait_ack(self, expected_seq: int, timeout: float = 1.0) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            pkt = self.read_packet()
            if pkt is None:
                continue
            if pkt["cmd"] == CMD_NECK_ACK and pkt["seq"] == expected_seq:
                return True
        return False

    def send_home(self):
        seq = self.send_packet(CMD_SET_NECK_HOME)
        ok = self.wait_ack(seq)
        print("HOME ACK:", ok)

    def send_yaw(self, yaw: int):
        payload = struct.pack("<h", yaw)
        seq = self.send_packet(CMD_SET_NECK_YAW, payload)
        ok = self.wait_ack(seq)
        print("YAW ACK:", ok, "yaw=", yaw)

    def send_lift(self, lift: int):
        payload = struct.pack("<h", lift)
        seq = self.send_packet(CMD_SET_NECK_LIFT, payload)
        ok = self.wait_ack(seq)
        print("LIFT ACK:", ok, "lift=", lift)

    def send_pitch_offset(self, pitch_offset: int):
        payload = struct.pack("<h", pitch_offset)
        seq = self.send_packet(CMD_SET_NECK_PITCH, payload)
        ok = self.wait_ack(seq)
        print("PITCH ACK:", ok, "pitchOffset=", pitch_offset)

    def send_pose(self, yaw: int, lift: int, pitch_offset: int):
        payload = struct.pack("<hhh", yaw, lift, pitch_offset)
        seq = self.send_packet(CMD_SET_NECK_POSE, payload)
        ok = self.wait_ack(seq)
        print("POSE ACK:", ok, "yaw=", yaw, "lift=", lift, "pitchOffset=", pitch_offset)

    def close(self):
        self.ser.close()


def main():
    client = Esp32NeckClient(PORT, BAUD)

    try:
        print("HOME")
        client.send_home()
        time.sleep(2.5)

        print("MID LIFT STRAIGHT")
        client.send_pose(385, 80, 0)
        time.sleep(3.0)

        print("MAX LIFT STRAIGHT")
        client.send_pose(385, 160, 0)
        time.sleep(3.0)

        print("LOOK BACK")
        client.send_pitch_offset(80)
        time.sleep(3.0)

        print("HOME")
        client.send_home()
        time.sleep(3.0)

        print("YAW LEFT")
        client.send_yaw(500)
        time.sleep(2.5)

        print("YAW RIGHT")
        client.send_yaw(300)
        time.sleep(2.5)

        print("HOME")
        client.send_home()
        time.sleep(2.5)

    finally:
        client.close()


if __name__ == "__main__":
    main()
