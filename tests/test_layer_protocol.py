from __future__ import annotations

import struct
import time

import pytest

from python.link import proto_layer as pl
from python.link.layer_host import LayerHost, LayerLinkError


class FakeLayerTransport:
    """Protocol-level fake. It validates host packets but is not an RTL simulator."""

    def __init__(self, *, wrong_seq=False, nack_cmd=None, drop_cmd=None, run_status=0):
        self.rx = bytearray()
        self.decoder = pl.Decoder()
        self.commands = []
        self.wrong_seq = wrong_seq
        self.nack_cmd = nack_cmd
        self.drop_cmd = drop_cmd
        self.run_status = run_status
        self.closed = False

    def clock(self):
        return time.monotonic()

    def write(self, data: bytes):
        frames = self.decoder.feed(data)
        assert len(frames) == 1 and isinstance(frames[0], pl.Frame)
        req = frames[0]
        self.commands.append(req)
        if req.cmd == self.drop_cmd:
            return

        seq = (req.seq + 1) & 0xFF if self.wrong_seq else req.seq
        if req.cmd == pl.CMD_INFO:
            payload = struct.pack(
                "<BBBBIIHHHHH", 2, 7, 7, 0x0F, 50_000_000, 921_600,
                4096, 16384, 1024, 64, 128,
            )
            rsp = pl.frame(pl.RSP_RESULT, seq, payload)
        elif req.cmd == self.nack_cmd:
            rsp = pl.frame(pl.RSP_NACK, seq, bytes((req.cmd, pl.NACK_BAD_ARGUMENT)))
        elif req.cmd == pl.CMD_RUN:
            rsp = pl.frame(pl.RSP_RESULT, seq, struct.pack("<BI", self.run_status, 321))
        elif req.cmd == pl.CMD_READ_FMAP:
            offset, count = struct.unpack("<HB", req.payload)
            value = 6 if offset == 0 else 0
            rsp = pl.frame(pl.RSP_RESULT, seq, bytes((value,)) * count)
        elif req.cmd == pl.CMD_READ_GAP:
            rsp = pl.frame(pl.RSP_RESULT, seq, bytes(128))
        elif req.cmd in pl.IMPLEMENTED_COMMANDS:
            rsp = pl.frame(pl.RSP_ACK, seq, bytes((req.cmd, 0)))
        else:
            rsp = pl.frame(pl.RSP_NACK, seq, bytes((req.cmd, pl.NACK_UNKNOWN_COMMAND)))
        self.rx.extend(rsp)

    def read(self, timeout: float) -> bytes:
        if self.rx:
            data = bytes(self.rx)
            self.rx.clear()
            return data
        # Avoid sleeping for the whole timeout in unit tests.
        time.sleep(min(timeout, 0.001))
        return b""

    def close(self):
        self.closed = True


def test_crc_check_value_and_frame_layout():
    assert pl.crc16(b"123456789") == 0x29B1
    packet = pl.frame(pl.CMD_INFO, 0x37, b"")
    assert packet[:6] == b"\xA5\x5A\x15\x37\x00\x00"
    assert int.from_bytes(packet[-2:], "little") == pl.crc16(packet[2:-2])


def test_decoder_handles_fragmentation_noise_and_multiple_frames():
    a = pl.frame(pl.CMD_INFO, 1, b"")
    b = pl.frame(pl.CMD_RUN, 2, b"")
    dec = pl.Decoder()
    assert dec.feed(b"\x00\x13" + a[:1]) == [pl.BadFrame("noise before A5 5A", b"\x00\x13")]
    assert dec.feed(a[1:] + b) == [pl.Frame(pl.CMD_INFO, 1, b""), pl.Frame(pl.CMD_RUN, 2, b"")]
    assert dec.buf == bytearray()


def test_decoder_rejects_crc_corruption():
    packet = bytearray(pl.frame(pl.CMD_INFO, 1, b""))
    packet[-1] ^= 0x80
    events = pl.Decoder().feed(bytes(packet))
    assert any(isinstance(event, pl.BadFrame) and event.reason == "CRC mismatch" for event in events)
    assert not any(isinstance(event, pl.Frame) for event in events)


def test_info_fields_are_decoded_from_22_byte_payload():
    payload = struct.pack("<BBBBIIHHHHH", 2, 7, 7, 0x0F, 50_000_000, 921_600,
                          4096, 16384, 1024, 64, 128)
    info = pl.decode_info(payload)
    assert (info.version, info.rows, info.cols) == (2, 7, 7)
    assert info.baud == 921_600 and info.max_payload == 128
    assert info.supports_out_int32 and info.supports_argmax
    assert info.supports_row_combine and info.supports_gap_in


def test_info_rejects_wrong_length_or_invalid_max_payload():
    with pytest.raises(ValueError, match="22 B"):
        pl.decode_info(b"\x00")
    payload = struct.pack("<BBBBIIHHHHH", 2, 7, 7, 0x0F, 50_000_000, 921_600,
                          4096, 16384, 1024, 64, 0)
    with pytest.raises(ValueError, match="MAX_PAYLOAD"):
        pl.decode_info(payload)


def test_command_payload_layouts_match_layer_controller():
    assert pl.encode_write_bytes(pl.CMD_WRITE_X, 0x1234, [-1, 2]) == b"\x34\x12\xff\x02"
    assert pl.encode_write_bias(3, [-1]) == b"\x03\x00\xff\xff\xff\xff"
    assert pl.encode_swap(x=True, w=False) == b"\x01"
    assert pl.encode_swap(x=False, w=True) == b"\x02"
    assert pl.encode_config(1, 1, 3, 1, False, False, 0) == struct.pack("<HHHHBBB", 1, 1, 3, 1, 0, 0, 0)
    assert len(pl.encode_config(1, 4, 68, 68, flags=0x0C)) == 12
    assert pl.encode_read_fmap(0x1234, 3) == b"\x34\x12\x03"


def test_write_encoder_rejects_invalid_values():
    with pytest.raises(ValueError):
        pl.encode_write_bytes(pl.CMD_WRITE_X, 0, [])
    with pytest.raises(ValueError):
        pl.encode_write_bytes(pl.CMD_WRITE_SHIFT, 0, [128])
    with pytest.raises(ValueError):
        pl.encode_config(1, 1, 3, 1, flags=0x10)
    with pytest.raises(ValueError):
        pl.encode_config(1, 1, 3, 1, pool=2)
    with pytest.raises(ValueError):
        pl.encode_read_fmap(0, 0)


def test_minimal_layer_smoke_command_sequence_and_result():
    transport = FakeLayerTransport()
    host = LayerHost(transport, timeout=0.1)
    result = host.smoke()
    assert result["passed"] is True
    assert result["got"] == result["expected"] == 6
    assert result["run"].cycles == 321
    assert [packet.cmd for packet in transport.commands] == [
        pl.CMD_INFO,
        pl.CMD_WRITE_X, pl.CMD_SWAP,
        pl.CMD_WRITE_W, pl.CMD_SWAP,
        pl.CMD_WRITE_BIAS, pl.CMD_WRITE_SHIFT,
        pl.CMD_CONFIG, pl.CMD_RUN, pl.CMD_READ_FMAP,
    ]
    assert [packet.seq for packet in transport.commands] == list(range(10))
    assert host.stats.transactions == 10


def test_nack_is_reported_without_retry():
    transport = FakeLayerTransport(nack_cmd=pl.CMD_WRITE_X)
    host = LayerHost(transport, timeout=0.1)
    host.info()
    with pytest.raises(LayerLinkError, match="NACK 2"):
        host.write_x(0, [1])
    assert [packet.cmd for packet in transport.commands].count(pl.CMD_WRITE_X) == 1


def test_wrong_sequence_is_rejected():
    transport = FakeLayerTransport(wrong_seq=True)
    host = LayerHost(transport, timeout=0.1)
    with pytest.raises(LayerLinkError, match="SEQ"):
        host.info()


def test_timeout_does_not_retry_a_command():
    transport = FakeLayerTransport(drop_cmd=pl.CMD_INFO)
    host = LayerHost(transport, timeout=0.015)
    with pytest.raises(LayerLinkError, match="not retried"):
        host.info()
    assert len(transport.commands) == 1


def test_pipeline_commands_are_documented_as_reserved_not_implemented():
    assert pl.RESERVED_COMMANDS == {0x11, 0x12, 0x13, 0x14}
    assert pl.CMD_INFO == 0x15
    assert not (pl.RESERVED_COMMANDS & pl.IMPLEMENTED_COMMANDS)


def test_host_chunks_memory_and_bias_writes_using_info_payload_limit():
    transport = FakeLayerTransport()
    host = LayerHost(transport, timeout=0.1)
    host.info()
    host.write_x(0, [0] * 300)
    host.write_bias(0, [0] * 64)
    x_packets = [p for p in transport.commands if p.cmd == pl.CMD_WRITE_X]
    bias_packets = [p for p in transport.commands if p.cmd == pl.CMD_WRITE_BIAS]
    assert [len(p.payload) for p in x_packets] == [128, 128, 50]
    assert [len(p.payload) for p in bias_packets] == [126, 126, 10]


def test_read_fmap_chunks_at_reported_max_payload():
    transport = FakeLayerTransport()
    host = LayerHost(transport, timeout=0.1)
    host.info()
    got = host.read_fmap(0, 260)
    reads = [p for p in transport.commands if p.cmd == pl.CMD_READ_FMAP]
    assert [len(p.payload) for p in reads] == [3, 3, 3]
    assert [p.payload[2] for p in reads] == [128, 128, 4]
    assert len(got) == 260


def test_run_returns_nonzero_status_for_runner_diagnostics():
    transport = FakeLayerTransport(run_status=1)
    host = LayerHost(transport, timeout=0.1)
    host.info()
    result = host.run()
    assert result.status == 1
    assert result.cycles == 321
