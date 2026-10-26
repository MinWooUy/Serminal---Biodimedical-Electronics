import io
import binascii
import struct
import unittest
from ecg_host import Accelerator, encode, receive


class FakePort(io.BytesIO):
    def __init__(self, response):
        super().__init__(response)
        self.sent = b""

    def write(self, data):
        self.sent += data
        return len(data)

    def flush(self):
        pass


class DuplexFakePort:
    """In-memory host/FPGA link: parse host frames and queue FPGA responses."""

    def __init__(self):
        self.sent = bytearray()
        self.received = bytearray()
        self.requests = []

    def write(self, data):
        self.sent.extend(data)
        self._respond(data)
        return len(data)

    def flush(self):
        pass

    def read(self, count):
        data = bytes(self.received[:count])
        del self.received[:count]
        return data

    def _respond(self, packet):
        self.assert_frame(packet)
        command, sequence, length = struct.unpack_from("<BBH", packet, 2)
        payload = packet[6:6 + length]
        self.requests.append((command, sequence, payload))
        if command == 4:
            response_payload = struct.pack("<9i", 1, 2, 3, 4, 5, 6, 7, 8, 9)
            response = encode(0x82, sequence, response_payload)
        elif command in (1, 2, 3, 5, 6):
            response = encode(0x80, sequence, bytes((command, 0)))
        else:
            response = encode(0x81, sequence, bytes((command, 4)))
        self.received.extend(response)

    @staticmethod
    def assert_frame(packet):
        if packet[:2] != b"\xA5\x5A":
            raise AssertionError("missing SOF")
        command, sequence, length = struct.unpack_from("<BBH", packet, 2)
        end = 6 + length
        if len(packet) != end + 2:
            raise AssertionError("invalid frame length")
        expected = binascii.crc_hqx(packet[2:end], 0xFFFF)
        if packet[end:] != struct.pack("<H", expected):
            raise AssertionError("invalid request CRC")


class HostTest(unittest.TestCase):
    def test_full_duplex_host_fpga_exchange(self):
        port = DuplexFakePort()
        accel = Accelerator(port)

        accel.write_tensor(1, [1, -2, 3, 4, -1])
        accel.write_tensor(2, [2, -1, 3, 1, 1, 1, -1, 2, 0])
        accel.configure(k=3, step=1, rows=3, cols=3)
        self.assertEqual(accel.run(), [(1, 2, 3), (4, 5, 6), (7, 8, 9)])

        self.assertEqual([request[0] for request in port.requests], [1, 2, 3, 4])
        self.assertEqual([request[1] for request in port.requests], [0, 1, 2, 3])
        self.assertEqual(port.received, bytearray())

    def test_encode_uses_little_endian_length_and_crc(self):
        payload = bytes(range(256))
        packet = encode(0x01, 0x2A, payload)
        self.assertEqual(packet[:6], b"\xA5\x5A\x01\x2A\x00\x01")
        body = packet[2:-2]
        expected_crc = binascii.crc_hqx(body, 0xFFFF)
        self.assertEqual(packet[-2:], struct.pack("<H", expected_crc))

    def test_receive_accepts_payload_containing_sync_bytes(self):
        payload = b"\xA5\x5A\x00\xFF"
        response = encode(0x80, 7, payload)
        command, sequence, received = receive(FakePort(b"garbage" + response))
        self.assertEqual((command, sequence, received), (0x80, 7, payload))

    def test_roundtrip_embedded_sof_and_signed_values(self):
        payload = struct.pack("<9i", -128, 127, -1, 0, 123456, -123456, 2, 3, 0x5AA5)
        port = FakePort(encode(0x82, 0, payload))
        accel = Accelerator(port)
        self.assertEqual(accel.run()[0], (-128, 127, -1))
        self.assertEqual(port.sent, encode(4, 0))

    def test_corruption(self):
        packet = bytearray(encode(0x80, 4, b"\x01\x00"))
        packet[6] ^= 1
        with self.assertRaisesRegex(ValueError, "CRC"):
            receive(FakePort(packet))

    def test_seq_mismatch(self):
        with self.assertRaisesRegex(ValueError, "sequence"):
            Accelerator(FakePort(encode(0x80, 1, b"\x05\x00"))).exchange(5)

    def test_nack(self):
        with self.assertRaisesRegex(RuntimeError, "NACK"):
            Accelerator(FakePort(encode(0x81, 0, b"\x04\x03"))).run()

    def test_quantization_range(self):
        with self.assertRaisesRegex(ValueError, "INT8"):
            Accelerator(FakePort(b"")).write_tensor(1, [128])

    def test_bias_preload_commands(self):
        port = DuplexFakePort()
        accel = Accelerator(port)
        accel.write_bias([11, -2, 7])
        accel.configure(k=3, step=1, rows=3, cols=3, preload_bias=True)
        self.assertEqual([request[0] for request in port.requests], [6, 3])
        self.assertEqual(port.requests[0][2], struct.pack("<Hiii", 0, 11, -2, 7))
        self.assertEqual(port.requests[1][2][-1], 1)

    def test_bias_range(self):
        with self.assertRaisesRegex(ValueError, "three output columns"):
            Accelerator(FakePort(b"")).write_bias([1], base=3)

    def test_timeout(self):
        with self.assertRaises(TimeoutError):
            receive(FakePort(b""), timeout=0.001)


if __name__ == "__main__":
    unittest.main()
