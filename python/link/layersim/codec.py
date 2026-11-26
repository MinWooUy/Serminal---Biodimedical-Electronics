"""Codec of the FPGA team's layer protocol (layer_cmd_controller.v, CMD 08..15; RTL Serminal dev ca7562e). Standard library only.

    A5 5A | CMD u8 | SEQ u8 | LEN u16 LE | PAYLOAD | CRC-16/CCITT-FALSE LE over CMD..PAYLOAD      (same frame as the tile protocol)

encode_cmd(cmd)        layersim/sched.py command -> (CMD, payload bytes)
frame(cmd, seq, body)  -> wire bytes
Decoder                bytes from the FPGA -> Ack / Nack / Result / BadFrame (host side; the FPGA side is fpga_sim.py)

A RESULT does not echo the CMD it answers (its payload depends on the command), so the host checks SEQ and the length it expects.
CMD 0x11..0x14 (CONFIG_ALL, LOAD_COMMIT, RUN_ALL, GET_RESULT) are reserved by the FPGA team and answered with NACK 4 today.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

from python.link.proto_v2 import crc16              # same CRC-16/CCITT-FALSE ("123456789" -> 0x29B1)

from . import sched as ls

SOF = b"\xA5\x5A"
HEADER_BYTES = 6                                     # SOF + CMD + SEQ + LEN
CRC_BYTES = 2
MAX_FRAME_PAYLOAD = 256                              # protocol limit; the bitstream's own limit is INFO.max_payload

WRITE_X, WRITE_W, WRITE_BIAS, WRITE_SHIFT, SWAP, CONFIG, RUN, READ_GAP, READ_FMAP = 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F, 0x10
CONFIG_ALL, LOAD_COMMIT, RUN_ALL, GET_RESULT = 0x11, 0x12, 0x13, 0x14
INFO = 0x15
ACK, NACK, RESULT = 0x80, 0x81, 0x82
NAMES = {WRITE_X: "WRITE_X", WRITE_W: "WRITE_W", WRITE_BIAS: "WRITE_BIAS", WRITE_SHIFT: "WRITE_SHIFT", SWAP: "SWAP",
         CONFIG: "CONFIG", RUN: "RUN", READ_GAP: "READ_GAP", READ_FMAP: "READ_FMAP", INFO: "INFO",
         CONFIG_ALL: "CONFIG_ALL", LOAD_COMMIT: "LOAD_COMMIT", RUN_ALL: "RUN_ALL", GET_RESULT: "GET_RESULT",
         ACK: "ACK", NACK: "NACK", RESULT: "RESULT"}
RESULT_CMDS = (RUN, READ_GAP, READ_FMAP, INFO)       # answered with RESULT; every other command with ACK
NACK_CRC, NACK_ARG, NACK_STATE, NACK_CMD = ls.NACK_CRC, ls.NACK_ARG, ls.NACK_STATE, ls.NACK_CMD
NACK_TEXT = {NACK_CRC: "khung hỏng (CRC sai hoặc LEN > MAX_PAYLOAD)", NACK_ARG: "tham số sai (độ dài / địa chỉ / cấu hình)",
             NACK_STATE: "sai trạng thái (RUN trước CONFIG, lớp đang chạy, hoặc READ_FMAP ngoài SRAM)", NACK_CMD: "lệnh lạ"}


def frame(cmd: int, seq: int, body: bytes = b"") -> bytes:
    if len(body) > MAX_FRAME_PAYLOAD:
        raise ValueError(f"payload {len(body)} B > {MAX_FRAME_PAYLOAD} B")
    head = struct.pack("<BBH", cmd, seq & 0xFF, len(body))
    return SOF + head + body + struct.pack("<H", crc16(head + body))


def _i8(values) -> bytes:
    if any(not -128 <= int(v) <= 127 for v in values):
        raise ValueError("int8 value out of range")
    return bytes(int(v) & 0xFF for v in values)


def encode_cmd(cmd) -> tuple[int, bytes]:
    if isinstance(cmd, ls.WriteX):
        return WRITE_X, struct.pack("<H", cmd.base) + _i8(cmd.data)
    if isinstance(cmd, ls.WriteW):
        return WRITE_W, struct.pack("<H", cmd.base) + _i8(cmd.data)
    if isinstance(cmd, ls.WriteShift):
        return WRITE_SHIFT, struct.pack("<H", cmd.base) + _i8(cmd.data)
    if isinstance(cmd, ls.WriteBias):
        return WRITE_BIAS, struct.pack("<H", cmd.base) + struct.pack(f"<{len(cmd.values)}i", *cmd.values)
    if isinstance(cmd, ls.Swap):
        return SWAP, bytes([(1 if cmd.x else 0) | (2 if cmd.w else 0)])
    if isinstance(cmd, ls.Config):
        body = struct.pack("<HHHHBBB", cmd.out_len, cmd.cout, cmd.k, cmd.step, cmd.pool, cmd.gap, cmd.pad_next)
        if cmd.w_base or cmd.b_base:                                              # 16 bytes: flags + w_base + b_base
            return CONFIG, body + bytes([cmd.flags]) + struct.pack("<HH", cmd.w_base, cmd.b_base)
        return CONFIG, body + (bytes([cmd.flags]) if cmd.flags else b"")      # 11 bytes = flags 0 (older hosts keep working)
    if isinstance(cmd, ls.Run):
        return RUN, b""
    if isinstance(cmd, ls.ReadGap):
        return READ_GAP, b""
    if isinstance(cmd, ls.ReadFmap):
        return READ_FMAP, struct.pack("<HB", cmd.offset, cmd.length)
    if isinstance(cmd, ls.Info):
        return INFO, b""
    raise TypeError(f"not a layer command: {cmd!r}")


def decode_cmd(code: int, p: bytes):
    """Inverse of encode_cmd (what the FPGA side parses). Raises ValueError where the RTL answers NACK 2."""
    n = len(p)
    if code in (WRITE_X, WRITE_W, WRITE_SHIFT):
        if n < 3:
            raise ValueError("length < 3")
        cls = {WRITE_X: ls.WriteX, WRITE_W: ls.WriteW, WRITE_SHIFT: ls.WriteShift}[code]
        return cls(struct.unpack("<H", p[:2])[0], tuple(v - 256 if v > 127 else v for v in p[2:]))
    if code == WRITE_BIAS:
        if n < 6 or n % 4 != 2:
            raise ValueError("length must be 2 + 4 * n")
        return ls.WriteBias(struct.unpack("<H", p[:2])[0], struct.unpack(f"<{(n - 2) // 4}i", p[2:]))
    if code == SWAP:
        if n != 1 or p[0] >> 2:
            raise ValueError("SWAP is one byte, bits 0..1")
        return ls.Swap(bool(p[0] & 1), bool(p[0] & 2))
    if code == CONFIG:
        if n not in (11, 12, 16):
            raise ValueError("CONFIG is 11, 12 or 16 bytes")
        out_len, cout, k, step, pool, gap, pad = struct.unpack("<HHHHBBB", p[:11])
        w_base, b_base = struct.unpack("<HH", p[12:16]) if n == 16 else (0, 0)
        return ls.Config(out_len, cout, k, step, pool, gap, pad, p[11] if n >= 12 else 0, w_base, b_base)
    if code in (RUN, READ_GAP, INFO):
        if n:
            raise ValueError("no payload expected")
        return {RUN: ls.Run("fpga"), READ_GAP: ls.ReadGap(), INFO: ls.Info()}[code]
    if code == READ_FMAP:
        if n != 3:
            raise ValueError("READ_FMAP is 3 bytes")
        off, ln = struct.unpack("<HB", p)
        return ls.ReadFmap(off, ln)
    raise KeyError(code)


# ---------------------------------------------------------------- answers (FPGA -> host)
@dataclass(frozen=True)
class Ack:
    seq: int
    cmd: int


@dataclass(frozen=True)
class Nack:
    seq: int
    cmd: int
    code: int


@dataclass(frozen=True)
class Result:
    seq: int
    data: bytes


@dataclass(frozen=True)
class BadFrame:
    why: str
    data: bytes


def parse_answer(cmd: int, seq: int, body: bytes):
    if cmd == ACK and len(body) == 2 and body[1] == 0:
        return Ack(seq, body[0])
    if cmd == NACK and len(body) == 2:
        return Nack(seq, body[0], body[1])
    if cmd == RESULT:
        return Result(seq, bytes(body))
    return BadFrame(f"unexpected answer CMD 0x{cmd:02X} with {len(body)} B", body)


class Decoder:
    """Host-side stream decoder: resynchronises on A5 5A; bad CRC / length -> BadFrame (the host reports, never guesses)."""

    def __init__(self, max_payload: int = MAX_FRAME_PAYLOAD):
        self.buf = bytearray()
        self.max_payload = max_payload

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
            if ln > self.max_payload:
                out.append(BadFrame(f"LEN {ln} > {self.max_payload}", bytes(self.buf[:HEADER_BYTES])))
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
