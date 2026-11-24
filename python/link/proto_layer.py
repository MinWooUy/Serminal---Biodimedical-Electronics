"""PC <-> FPGA layer-level UART protocol used by rtl/cnn1d/top/layer_cmd_controller.v.

Frame:
    A5 5A | CMD:u8 | SEQ:u8 | LEN:u16 LE | PAYLOAD | CRC16:u16 LE

CRC-16/CCITT-FALSE covers CMD..PAYLOAD.  This module intentionally does not
import tile_sched: the layer protocol is a different command contract.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Iterable

try:
    from python.link.proto_v2 import crc16
except ImportError:  # pragma: no cover - stand-alone copy beside proto_v2.py
    from proto_v2 import crc16

SOF = b"\xA5\x5A"
HEADER_BYTES = 6
CRC_BYTES = 2
MAX_PAYLOAD = 256
BIAS_DEPTH = 256                 # layer_cmd_controller.v BIAS_DEPTH (not reported by INFO); bias RAM entries (INT32)

CMD_WRITE_X = 0x08
CMD_WRITE_W = 0x09
CMD_WRITE_BIAS = 0x0A
CMD_WRITE_SHIFT = 0x0B
CMD_SWAP = 0x0C
CMD_CONFIG = 0x0D
CMD_RUN = 0x0E
CMD_READ_GAP = 0x0F
CMD_READ_FMAP = 0x10
CMD_CONFIG_ALL = 0x11       # reserved in current RTL FSM
CMD_LOAD_COMMIT = 0x12      # reserved in current RTL FSM
CMD_RUN_ALL = 0x13          # reserved in current RTL FSM
CMD_GET_RESULT = 0x14       # reserved in current RTL FSM
CMD_INFO = 0x15

CMD_NAMES = {
    CMD_WRITE_X: "WRITE_X",
    CMD_WRITE_W: "WRITE_W",
    CMD_WRITE_BIAS: "WRITE_BIAS",
    CMD_WRITE_SHIFT: "WRITE_SHIFT",
    CMD_SWAP: "SWAP",
    CMD_CONFIG: "CONFIG",
    CMD_RUN: "RUN",
    CMD_READ_GAP: "READ_GAP",
    CMD_READ_FMAP: "READ_FMAP",
    CMD_CONFIG_ALL: "CONFIG_ALL",
    CMD_LOAD_COMMIT: "LOAD_COMMIT",
    CMD_RUN_ALL: "RUN_ALL",
    CMD_GET_RESULT: "GET_RESULT",
    CMD_INFO: "INFO",
}
IMPLEMENTED_COMMANDS = frozenset((
    CMD_WRITE_X, CMD_WRITE_W, CMD_WRITE_BIAS, CMD_WRITE_SHIFT, CMD_SWAP,
    CMD_CONFIG, CMD_RUN, CMD_READ_GAP, CMD_READ_FMAP, CMD_INFO,
))
RESERVED_COMMANDS = frozenset((CMD_CONFIG_ALL, CMD_LOAD_COMMIT, CMD_RUN_ALL, CMD_GET_RESULT))

RSP_ACK = 0x80
RSP_NACK = 0x81
RSP_RESULT = 0x82
RSP_NAMES = {RSP_ACK: "ACK", RSP_NACK: "NACK", RSP_RESULT: "RESULT"}

NACK_BAD_FRAME = 1
NACK_BAD_ARGUMENT = 2
NACK_BAD_STATE = 3
NACK_UNKNOWN_COMMAND = 4
NACK_TEXT = {
    NACK_BAD_FRAME: "frame/CRC error",
    NACK_BAD_ARGUMENT: "bad length, address, or configuration",
    NACK_BAD_STATE: "command not valid in the current state / missing data",
    NACK_UNKNOWN_COMMAND: "unknown command (reserved or unsupported by this RTL)",
}


@dataclass(frozen=True)
class Frame:
    cmd: int
    seq: int
    payload: bytes


@dataclass(frozen=True)
class BadFrame:
    reason: str
    data: bytes


@dataclass(frozen=True)
class HardwareInfo:
    version: int
    rows: int
    cols: int
    features: int
    clk_hz: int
    baud: int
    ifmap_depth: int
    wgt_depth: int
    feeder_depth: int
    cout_max: int
    max_payload: int

    @property
    def supports_out_int32(self) -> bool:
        return bool(self.features & 0x01)

    @property
    def supports_argmax(self) -> bool:
        return bool(self.features & 0x02)

    @property
    def supports_row_combine(self) -> bool:
        return bool(self.features & 0x04)

    @property
    def supports_gap_in(self) -> bool:
        return bool(self.features & 0x08)

    @property
    def supports_resident(self) -> bool:
        """INFO features bit 4: CONFIG may carry w_base / b_base, so weights and biases can stay on the chip."""
        return bool(self.features & 0x10)


def frame(cmd: int, seq: int, payload: bytes = b"", max_payload: int = MAX_PAYLOAD) -> bytes:
    """Encode one packet. `max_payload` is the limit reported by the target FPGA."""
    if not 0 <= int(cmd) <= 0xFF:
        raise ValueError("CMD must fit in u8")
    if not 0 <= int(seq) <= 0xFF:
        raise ValueError("SEQ must fit in u8")
    payload = bytes(payload)
    if len(payload) > min(int(max_payload), MAX_PAYLOAD):
        raise ValueError(f"payload {len(payload)} B exceeds limit {min(int(max_payload), MAX_PAYLOAD)} B")
    header = struct.pack("<BBH", cmd, seq, len(payload))
    return SOF + header + payload + struct.pack("<H", crc16(header + payload))


class Decoder:
    """Incremental byte-stream decoder for layer protocol frames."""

    def __init__(self, max_payload: int = MAX_PAYLOAD):
        if not 1 <= int(max_payload) <= MAX_PAYLOAD:
            raise ValueError("max_payload must be in 1..256")
        self.max_payload = int(max_payload)
        self.buf = bytearray()

    def set_max_payload(self, max_payload: int) -> None:
        if not 1 <= int(max_payload) <= MAX_PAYLOAD:
            raise ValueError("max_payload must be in 1..256")
        self.max_payload = int(max_payload)

    def feed(self, data: bytes) -> list[Frame | BadFrame]:
        self.buf.extend(data)
        out: list[Frame | BadFrame] = []
        while True:
            start = self.buf.find(SOF)
            if start < 0:
                keep = 1 if self.buf[-1:] == SOF[:1] else 0
                n_drop = len(self.buf) - keep
                if n_drop:
                    out.append(BadFrame("noise before A5 5A", bytes(self.buf[:n_drop])))
                    del self.buf[:n_drop]
                return out
            if start:
                out.append(BadFrame("noise before A5 5A", bytes(self.buf[:start])))
                del self.buf[:start]
            if len(self.buf) < HEADER_BYTES:
                return out

            cmd, seq, length = struct.unpack_from("<BBH", self.buf, 2)
            if length > self.max_payload:
                out.append(BadFrame(f"LEN {length} > MAX_PAYLOAD {self.max_payload}", bytes(self.buf[:HEADER_BYTES])))
                # The header may itself be corrupted; discard one byte and search again.
                del self.buf[:1]
                continue

            total = HEADER_BYTES + length + CRC_BYTES
            if len(self.buf) < total:
                return out
            payload = bytes(self.buf[HEADER_BYTES:HEADER_BYTES + length])
            got_crc = struct.unpack_from("<H", self.buf, total - CRC_BYTES)[0]
            expected_crc = crc16(bytes(self.buf[2:HEADER_BYTES]) + payload)
            if got_crc != expected_crc:
                out.append(BadFrame("CRC mismatch", bytes(self.buf[:total])))
                del self.buf[:1]
                continue

            out.append(Frame(cmd, seq, payload))
            del self.buf[:total]

    def reset(self) -> None:
        self.buf.clear()


def _check_u16(name: str, value: int) -> int:
    value = int(value)
    if not 0 <= value <= 0xFFFF:
        raise ValueError(f"{name} must fit in u16")
    return value


def _check_u8(name: str, value: int) -> int:
    value = int(value)
    if not 0 <= value <= 0xFF:
        raise ValueError(f"{name} must fit in u8")
    return value


def _int8_bytes(values: Iterable[int]) -> bytes:
    out = []
    for value in values:
        value = int(value)
        if not -128 <= value <= 127:
            raise ValueError(f"int8 value {value} out of range")
        out.append(value & 0xFF)
    return bytes(out)


def encode_write_bytes(cmd: int, address: int, values: Iterable[int]) -> bytes:
    if cmd not in (CMD_WRITE_X, CMD_WRITE_W, CMD_WRITE_SHIFT):
        raise ValueError(f"CMD 0x{cmd:02X} is not an INT8 write command")
    data = _int8_bytes(values)
    if not data:
        raise ValueError("write command must contain at least one data byte")
    return struct.pack("<H", _check_u16("address", address)) + data


def encode_write_bias(base: int, values: Iterable[int]) -> bytes:
    vals = tuple(int(v) for v in values)
    if not vals:
        raise ValueError("WRITE_BIAS requires at least one INT32 value")
    if any(v < -(1 << 31) or v >= (1 << 31) for v in vals):
        raise ValueError("bias value outside signed INT32 range")
    return struct.pack("<H", _check_u16("base", base)) + struct.pack(f"<{len(vals)}i", *vals)


def encode_swap(x: bool = False, w: bool = False) -> bytes:
    flags = (1 if x else 0) | (2 if w else 0)
    if flags == 0:
        raise ValueError("SWAP must select IFMap (x), Weight (w), or both")
    return bytes((flags,))


def encode_config(
    out_len: int,
    cout: int,
    k_len: int,
    window_step: int,
    pool: bool = False,
    gap: bool = False,
    pad_next: int = 0,
    flags: int | None = None,
    w_base: int | None = None,
    b_base: int | None = None,
) -> bytes:
    """Encode 11-byte CONFIG, 12 bytes with feature flags, or 16 bytes with flags + w_base + b_base (INFO features bit 4).

    w_base: start of this layer's weights inside the READ weight bank; b_base: start of its biases in the bias RAM.
    Without them (11 / 12 bytes) the FPGA reads both from address 0."""
    vals16 = tuple(_check_u16(n, v) for n, v in (
        ("out_len", out_len), ("cout", cout), ("k_len", k_len), ("window_step", window_step)
    ))
    pool_v = _check_u8("pool", int(pool))
    gap_v = _check_u8("gap", int(gap))
    if pool_v > 1 or gap_v > 1:
        raise ValueError("pool and gap must be 0 or 1")
    vals8 = (pool_v, gap_v, _check_u8("pad_next", pad_next))
    payload = struct.pack("<HHHHBBB", *vals16, *vals8)
    if flags is not None:
        flags = _check_u8("flags", flags)
        if flags & 0xF0:
            raise ValueError("CONFIG feature flags may only use bits 0..3")
        payload += bytes((flags,))
    if w_base is not None or b_base is not None:
        if flags is None:
            payload += bytes((0,))                      # the 16-byte form always carries the flags byte
        payload += struct.pack("<HH", _check_u16("w_base", w_base or 0), _check_u16("b_base", b_base or 0))
    return payload


def encode_read_fmap(offset: int, length: int) -> bytes:
    offset = _check_u16("offset", offset)
    length = int(length)
    if not 1 <= length <= 255:
        raise ValueError("READ_FMAP length must be in 1..255")
    return struct.pack("<HB", offset, length)


def decode_info(payload: bytes) -> HardwareInfo:
    """Decode the 22-byte INFO result defined by layer_cmd_controller.v (version 2)."""
    if len(payload) != 22:
        raise ValueError(f"INFO payload must be 22 B, received {len(payload)} B")
    fields = struct.unpack("<BBBBIIHHHHH", payload)
    info = HardwareInfo(*fields)
    if info.version < 1:
        raise ValueError(f"unsupported INFO version {info.version}")
    if not (1 <= info.rows <= 255 and 1 <= info.cols <= 255):
        raise ValueError("invalid PE array dimensions reported by FPGA")
    if not (1 <= info.max_payload <= MAX_PAYLOAD):
        raise ValueError(f"invalid MAX_PAYLOAD reported by FPGA: {info.max_payload}")
    if info.baud <= 0 or info.clk_hz <= 0:
        raise ValueError("invalid clock/baud reported by FPGA")
    return info
