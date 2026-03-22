import serial
import sys
import time

SOF1 = 0xAA
SOF2 = 0x55

CMD_SET_HEAD_MODE = 0x30
CMD_SET_HEAD_BRIGHTNESS = 0x32
CMD_SET_HEAD_LEVEL = 0x33
CMD_SET_HEAD_SPEAK_STYLE = 0x34

def send_packet(ser, cmd, seq, payload):
    pkt = bytearray()
    pkt.append(SOF1)
    pkt.append(SOF2)
    pkt.append(len(payload) + 2)   # CMD + SEQ + PAYLOAD
    pkt.append(cmd)
    pkt.append(seq)
    pkt.extend(payload)
    ser.write(pkt)

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python3 send_head_mode.py mode <value>")
        print("  python3 send_head_mode.py brightness <value>")
        print("  python3 send_head_mode.py level <value>")
        return

    ser = serial.Serial("/dev/ttyTHS1", 115200, timeout=0.2)
    time.sleep(0.1)

    kind = sys.argv[1]
    value = int(sys.argv[2])

    if kind == "mode":
        send_packet(ser, CMD_SET_HEAD_MODE, 1, bytes([value]))
    elif kind == "brightness":
        send_packet(ser, CMD_SET_HEAD_BRIGHTNESS, 1, bytes([value]))
    elif kind == "level":
        send_packet(ser, CMD_SET_HEAD_LEVEL, 1, bytes([value]))
    elif kind == "style":
    	send_packet(ser, CMD_SET_HEAD_SPEAK_STYLE, 1, bytes([value]))
    else:
        print("Unknown command type")
        return

    print(f"Sent {kind}={value}")
    ser.close()

if __name__ == "__main__":
    main()
