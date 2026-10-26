#!/usr/bin/env python3
"""Stop-and-wait client for ecg_accel_top. Requires pyserial for hardware I/O."""
import argparse
import binascii
import struct
import time

SOF = b"\xa5\x5a"
MAX_PAYLOAD = 256


def encode(cmd, seq, payload=b""):
    if len(payload) > MAX_PAYLOAD:
        raise ValueError("Payload exceeds 256 bytes")
    body = struct.pack("<BBH", cmd, seq, len(payload)) + payload
    return SOF + body + struct.pack("<H", binascii.crc_hqx(body, 0xFFFF))


def read_exact(port, count, deadline):
    data = bytearray()
    while len(data) < count:
        if time.monotonic() >= deadline:
            raise TimeoutError("FPGA response timed out; do not blindly retry RUN")
        chunk = port.read(count - len(data))
        data.extend(chunk)
    return bytes(data)


def receive(port, timeout=5.0):
    deadline = time.monotonic() + timeout
    previous = b""
    while True:
        byte = read_exact(port, 1, deadline)
        if previous + byte == SOF:
            break
        previous = byte
    header = read_exact(port, 4, deadline)
    cmd, seq, length = struct.unpack("<BBH", header)
    if length > MAX_PAYLOAD:
        raise ValueError("Oversized response")
    payload = read_exact(port, length, deadline)
    received_crc, = struct.unpack("<H", read_exact(port, 2, deadline))
    if received_crc != binascii.crc_hqx(header + payload, 0xFFFF):
        raise ValueError("Response CRC mismatch")
    return cmd, seq, payload


class Accelerator:
    def __init__(self, port):
        self.port = port
        self.seq = 0

    def exchange(self, cmd, payload=b""):
        seq = self.seq
        self.seq = (seq + 1) & 255
        self.port.write(encode(cmd, seq, payload))
        self.port.flush()
        response, returned_seq, data = receive(self.port)
        if returned_seq != seq:
            raise ValueError("Response sequence mismatch")
        if response == 0x81:
            raise RuntimeError(f"FPGA NACK: {data.hex()}")
        expected = 0x82 if cmd == 4 else 0x80
        if response != expected:
            raise ValueError("Unexpected response command")
        if response == 0x80 and (len(data) < 2 or data[:2] != bytes([cmd, 0])):
            raise ValueError("Malformed ACK")
        return data

    def write_tensor(self, cmd, values, base=0):
        if cmd not in (1, 2):
            raise ValueError("Tensor command must be 1 or 2")
        if any(v < -128 or v > 127 for v in values):
            raise ValueError("Tensor must contain signed INT8 values")
        if base < 0 or base + len(values) > 1024:
            raise ValueError("Tensor exceeds default 1024-byte bank")
        data = bytes(v & 255 for v in values)
        for offset in range(0, len(data), MAX_PAYLOAD - 2):
            self.exchange(cmd, struct.pack("<H", base + offset) + data[offset:offset + 254])

    def write_bias(self, values, base=0):
        if not values or any(v < -(2**31) or v > 2**31 - 1 for v in values):
            raise ValueError("Biases must be one or more signed INT32 values")
        if base < 0 or base + len(values) > 3:
            raise ValueError("Bias exceeds the three output columns of the default tile")
        self.exchange(6, struct.pack("<H", base) + struct.pack(f"<{len(values)}i", *values))

    def configure(self, k, step, rows, cols, x_base=0, w_base=0, preload_bias=False):
        if not (1 <= k <= 16 and 1 <= rows <= 3 and 1 <= cols <= 3 and step > 0):
            raise ValueError("Invalid default tile configuration")
        payload = struct.pack("<HHBBHH", k, step, rows, cols, x_base, w_base)
        # The 10-byte payload remains compatible with existing FPGA images.
        if preload_bias:
            payload += b"\x01"
        self.exchange(3, payload)

    def run(self):
        data = self.exchange(4)
        if len(data) != 36:
            raise ValueError("Expected nine INT32 results")
        values = struct.unpack("<9i", data)
        return [values[i:i+3] for i in range(0, 9, 3)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("port", help="Serial port, e.g. /dev/ttyUSB0 or COM3")
    parser.add_argument("--baud", type=int, default=115200)
    args = parser.parse_args()
    import serial
    with serial.Serial(args.port, args.baud, timeout=0.1, write_timeout=5) as port:
        accel = Accelerator(port)
        # One input channel, kernel width 3, stride 1, three output positions.
        accel.write_tensor(1, [1, -2, 3, 4, -1])
        accel.write_tensor(2, [2, -1, 3, 1, 1, 1, -1, 2, 0])
        accel.configure(k=3, step=1, rows=3, cols=3)
        result = accel.run()
        expected = [(13, 2, -5), (5, 5, 8), (-1, 6, 5)]
        print("Result:", result)
        if result != expected:
            raise RuntimeError(f"Expected {expected}")
        print("PASS")


if __name__ == "__main__":
    main()
