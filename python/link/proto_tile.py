"""Codec of the FPGA team's tile protocol (docs/proto_tile.md; RTL Serminal dev 4ad6c8b). Standard library only.

    A5 5A | CMD u8 | SEQ u8 | LEN u16 LE | PAYLOAD | CRC-16/CCITT-FALSE LE over CMD..PAYLOAD

encode_cmd(cmd)        python/link/tile_sched.py command -> (CMD, payload bytes)
frame(cmd, seq, body)  -> wire bytes
Decoder                bytes from the FPGA -> Ack / Nack / Result / BadFrame (host side; the FPGA side is tile_fpga_sim.py)
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

try:
    from python.link.proto_v2 import crc16      # same CRC-16/CCITT-FALSE ("123456789" -> 0x29B1)
    from python.link import tile_sched as ts
except ImportError:                             # stand-alone copy in the hand-off package
    from proto_v2 import crc16
    import tile_sched as ts

SOF = b"\xA5\x5A"
HEADER_BYTES = 6                                 # SOF + CMD + SEQ + LEN
CRC_BYTES = 2

WRITE_IFMAP, WRITE_WEIGHT, CONFIG_TILE, RUN_TILE, GET_STATUS, WRITE_BIAS, CONFIG_POST = 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07
ACK, NACK, RESULT = 0x80, 0x81, 0x82
NAMES = {WRITE_IFMAP: "WRITE_IFMAP", WRITE_WEIGHT: "WRITE_WEIGHT", CONFIG_TILE: "CONFIG_TILE", RUN_TILE: "RUN_TILE",
         GET_STATUS: "GET_STATUS", WRITE_BIAS: "WRITE_BIAS", CONFIG_POST: "CONFIG_POST", ACK: "ACK", NACK: "NACK",
         RESULT: "RESULT"}
NACK_CRC, NACK_LEN, NACK_NOT_READY, NACK_CMD = 1, 2, 3, 4
NACK_TEXT = {NACK_CRC: "CRC sai hoặc LEN > 256", NACK_LEN: "sai độ dài / cấu hình / địa chỉ",
             NACK_NOT_READY: "chưa chạy được / thiếu dữ liệu", NACK_CMD: "CMD lạ"}


class GetStatus:
    """GET_STATUS has no tile_sched counterpart (the planner never needs it)."""


def frame(cmd: int, seq: int, body: bytes = b"") -> bytes:
    if len(body) > ts.MAX_PAYLOAD:
        raise ValueError(f"payload {len(body)} B > {ts.MAX_PAYLOAD} B")
    head = struct.pack("<BBH", cmd, seq & 0xFF, len(body))
    return SOF + head + body + struct.pack("<H", crc16(head + body))


def _i8(values) -> bytes:
    if any(not -128 <= int(v) <= 127 for v in values):
        raise ValueError("int8 value out of range")
    return bytes(int(v) & 0xFF for v in values)


def encode_cmd(cmd) -> tuple[int, bytes]:
    if isinstance(cmd, ts.WriteIfmap):
        return WRITE_IFMAP, struct.pack("<H", cmd.base) + _i8(cmd.data)
    if isinstance(cmd, ts.WriteWeight):
        return WRITE_WEIGHT, struct.pack("<H", cmd.base) + _i8(cmd.data)
    if isinstance(cmd, ts.WriteBias):
        return WRITE_BIAS, struct.pack("<H", cmd.base) + struct.pack(f"<{len(cmd.values)}i", *cmd.values)
    if isinstance(cmd, ts.ConfigPost):
        return CONFIG_POST, _i8(cmd.shifts) + bytes([1 if cmd.relu else 0])
    if isinstance(cmd, ts.ConfigTile):
        body = struct.pack("<HHBBHH", cmd.k, cmd.step, cmd.rows, cmd.cols, cmd.x_base, cmd.w_base)
        return CONFIG_TILE, body + (b"\x01" if cmd.preload else b"")
    if isinstance(cmd, ts.RunTile):
        return RUN_TILE, b""
    if isinstance(cmd, GetStatus):
        return GET_STATUS, b""
    raise TypeError(f"not a tile command: {cmd!r}")


# ---------------------------------------------------------------- answers (FPGA -> host)
@dataclass(frozen=True)
class Ack:
    seq: int
    cmd: int
    flags: int | None = None          # GET_STATUS only


@dataclass(frozen=True)
class Nack:
    seq: int
    cmd: int
    code: int


@dataclass(frozen=True)
class Result:
    seq: int
    raw: tuple                        # ROWS*COLS int32, row-major (row*COLS + col)
    post: tuple | None                # ROWS*COLS int8 when POST_ENABLE


@dataclass(frozen=True)
class BadFrame:
    why: str
    data: bytes


def parse_answer(cmd: int, seq: int, body: bytes):
    n = ts.ROWS * ts.COLS
    if cmd == ACK and len(body) in (2, 3) and body[1] == 0:
        return Ack(seq, body[0], body[2] if len(body) == 3 else None)
    if cmd == NACK and len(body) == 2:
        return Nack(seq, body[0], body[1])
    if cmd == RESULT and len(body) in (4 * n, 5 * n):
        raw = struct.unpack(f"<{n}i", body[:4 * n])
        post = struct.unpack(f"<{n}b", body[4 * n:]) if len(body) == 5 * n else None
        return Result(seq, raw, post)
    return BadFrame(f"unexpected answer CMD 0x{cmd:02X} with {len(body)} B", body)


class Decoder:
    """Host-side stream decoder: resynchronises on A5 5A; bad CRC / length -> BadFrame (the host reports, never guesses)."""

    def __init__(self):
        self.buf = bytearray()

    def feed(self, data: bytes) -> list:
        self.buf += data
        out = []
        while True:
            k = self.buf.find(SOF)
            if k < 0:
                keep = 1 if self.buf[-1:] == SOF[:1] else 0
                if len(self.buf) > keep:
                    out.append(BadFrame("bytes before A5 5A", bytes(self.buf[:len(self.buf) - keep])))
                    del self.buf[:len(self.buf) - keep]
                return out
            if k:
                out.append(BadFrame("bytes before A5 5A", bytes(self.buf[:k])))
                del self.buf[:k]
            if len(self.buf) < HEADER_BYTES:
                return out
            cmd, seq, ln = struct.unpack("<BBH", self.buf[2:HEADER_BYTES])
            if ln > ts.MAX_PAYLOAD:
                out.append(BadFrame(f"LEN {ln} > {ts.MAX_PAYLOAD}", bytes(self.buf[:HEADER_BYTES])))
                del self.buf[:2]
                continue
            end = HEADER_BYTES + ln + CRC_BYTES
            if len(self.buf) < end:
                return out
            body = bytes(self.buf[HEADER_BYTES:HEADER_BYTES + ln])
            got = struct.unpack("<H", self.buf[end - 2:end])[0]
            if crc16(bytes(self.buf[2:HEADER_BYTES]) + body) != got:
                out.append(BadFrame("CRC", bytes(self.buf[:end])))
                del self.buf[:2]
                continue
            out.append(parse_answer(cmd, seq, body))
            del self.buf[:end]
