"""PC <-> FPGA link, protocol version 2 (CNN-only accelerator). Spec: docs/proto_v2.md.

Self-contained (standard library only) so that it can be shipped in the FPGA hand-off package.

Frame:  0xA5 0x5A | TYPE u8 | LEN u16 LE | PAYLOAD (LEN bytes) | CRC u16 LE
        CRC-16/CCITT-FALSE (poly 0x1021, init 0xFFFF, no reflection, xorout 0) over TYPE + LEN + PAYLOAD;
        check("123456789") = 0x29B1. All multi-byte fields little-endian, signed fields two's complement.

  0x01 PING     (host -> FPGA)  0 B
  0x02 PONG     (FPGA -> host)  proto_ver u8, mode u8, build_id u32, weights_crc32 u32          10 B
  0x10 BEAT_IN  (host -> FPGA)  beat_id u32, z_q i8 x 360, rr_q i8 x 4                          368 B
  0x11 RESULT   (FPGA -> host)  beat_id u32, logit i32 x 4, pred u8, cycles u32                 25 B
  0x1F NACK     (FPGA -> host)  reason u8, last_type u8, err_count u16                          4 B

NACK reasons: 1 bad CRC, 2 LEN not valid for TYPE, 3 unknown TYPE, 4 busy.

Decoder: a byte-stream state machine. After the sync word it checks TYPE (must be one the receiving side
accepts) and LEN (must equal the fixed length of TYPE) BEFORE waiting for the payload, so a corrupted LEN can
never stall it; then the CRC. Any rejection is reported as a FrameError (reason 1/2/3, type byte) and ONE
byte is discarded before scanning again for 0xA5 0x5A.
"""

import struct
import zlib
from dataclasses import dataclass, field

PROTO_VER = 2
MODE_CNN_ONLY = 1
SYNC = b"\xa5\x5a"
HEADER_BYTES = 5                       # sync (2) + type (1) + len (2)
CRC_BYTES = 2

T_PING, T_PONG, T_BEAT_IN, T_RESULT, T_NACK = 0x01, 0x02, 0x10, 0x11, 0x1F
N_Z, N_RR, N_CLASSES = 360, 4, 4

NACK_BAD_CRC, NACK_BAD_LEN, NACK_BAD_TYPE, NACK_BUSY = 1, 2, 3, 4

_PONG = struct.Struct("<BBII")
_BEAT_IN = struct.Struct(f"<I{N_Z}b{N_RR}b")
_RESULT = struct.Struct(f"<I{N_CLASSES}iBI")
_NACK = struct.Struct("<BBH")

PAYLOAD_LEN = {T_PING: 0, T_PONG: _PONG.size, T_BEAT_IN: _BEAT_IN.size, T_RESULT: _RESULT.size, T_NACK: _NACK.size}
HOST_TO_FPGA = frozenset((T_PING, T_BEAT_IN))
FPGA_TO_HOST = frozenset((T_PONG, T_RESULT, T_NACK))

# timing of the stop-and-wait host (seconds) and retransmissions after the first send
PONG_TIMEOUT_S, RESULT_TIMEOUT_S, MAX_RETRIES = 0.5, 1.0, 3


def _crc16_table(poly: int = 0x1021) -> tuple:
    t = []
    for b in range(256):
        c = b << 8
        for _ in range(8):
            c = ((c << 1) ^ poly) & 0xFFFF if c & 0x8000 else (c << 1) & 0xFFFF
        t.append(c)
    return tuple(t)


_CRC16_TABLE = _crc16_table()


def crc16(data: bytes, crc: int = 0xFFFF) -> int:
    """CRC-16/CCITT-FALSE: poly 0x1021, init 0xFFFF, MSB first, no final XOR."""
    for b in data:
        crc = ((crc << 8) & 0xFFFF) ^ _CRC16_TABLE[(crc >> 8) ^ b]
    return crc


def crc32(data: bytes, crc: int = 0) -> int:
    """CRC-32/IEEE (zlib); chained: crc32(b, crc32(a)) == crc32(a + b)."""
    return zlib.crc32(data, crc) & 0xFFFFFFFF


# ---------------------------------------------------------------- packets

@dataclass(frozen=True)
class Ping:
    TYPE = T_PING


@dataclass(frozen=True)
class Pong:
    proto_ver: int
    mode: int
    build_id: int
    weights_crc32: int
    TYPE = T_PONG


@dataclass(frozen=True)
class BeatIn:
    beat_id: int
    z_q: tuple                     # 360 ints in [-128, 127]
    rr_q: tuple                    # 4 ints in [-128, 127]
    TYPE = T_BEAT_IN


@dataclass(frozen=True)
class Result:
    beat_id: int
    logits: tuple                  # 4 int32
    pred: int
    cycles: int
    TYPE = T_RESULT


@dataclass(frozen=True)
class Nack:
    reason: int
    last_type: int
    err_count: int
    TYPE = T_NACK


@dataclass(frozen=True)
class FrameError:
    """A rejected frame (decoder event, not a packet): reason = NACK reason code, ftype = TYPE byte seen."""
    reason: int
    ftype: int


def payload(pkt) -> bytes:
    if isinstance(pkt, Ping):
        return b""
    if isinstance(pkt, Pong):
        return _PONG.pack(pkt.proto_ver, pkt.mode, pkt.build_id, pkt.weights_crc32)
    if isinstance(pkt, BeatIn):
        if len(pkt.z_q) != N_Z or len(pkt.rr_q) != N_RR:
            raise ValueError("BEAT_IN needs 360 z_q and 4 rr_q values")
        return _BEAT_IN.pack(pkt.beat_id, *pkt.z_q, *pkt.rr_q)
    if isinstance(pkt, Result):
        return _RESULT.pack(pkt.beat_id, *pkt.logits, pkt.pred, pkt.cycles)
    if isinstance(pkt, Nack):
        return _NACK.pack(pkt.reason, pkt.last_type, pkt.err_count)
    raise TypeError(f"not a protocol v2 packet: {pkt!r}")


def frame(ftype: int, body: bytes) -> bytes:
    if len(body) > 0xFFFF:
        raise ValueError("payload longer than 65535 bytes")
    head = struct.pack("<BH", ftype, len(body))
    return SYNC + head + body + struct.pack("<H", crc16(head + body))


def encode(pkt) -> bytes:
    return frame(pkt.TYPE, payload(pkt))


def parse(ftype: int, p: bytes):
    if ftype == T_PING:
        return Ping()
    if ftype == T_PONG:
        return Pong(*_PONG.unpack(p))
    if ftype == T_BEAT_IN:
        v = _BEAT_IN.unpack(p)
        return BeatIn(v[0], v[1:1 + N_Z], v[1 + N_Z:])
    if ftype == T_RESULT:
        v = _RESULT.unpack(p)
        return Result(v[0], v[1:5], v[5], v[6])
    if ftype == T_NACK:
        return Nack(*_NACK.unpack(p))
    raise ValueError(f"unknown type 0x{ftype:02X}")


# ---------------------------------------------------------------- decoder

@dataclass
class Decoder:
    """Feed arbitrary byte chunks; returns packets and FrameError events in stream order.

    accept: TYPE codes this side accepts (default: every protocol v2 type). A TYPE outside `accept` is
    reported as reason 3 (unknown TYPE)."""
    accept: frozenset = frozenset(PAYLOAD_LEN)
    buf: bytearray = field(default_factory=bytearray)
    frames_ok: int = 0
    bad_crc: int = 0
    bad_len: int = 0
    bad_type: int = 0
    skipped_bytes: int = 0

    @property
    def frame_errors(self) -> int:
        return self.bad_crc + self.bad_len + self.bad_type

    def feed(self, data: bytes) -> list:
        self.buf += data
        out = []
        while True:
            k = self.buf.find(SYNC)
            if k < 0:
                keep = 1 if self.buf[-1:] == SYNC[:1] else 0        # a split sync word may continue
                self.skipped_bytes += len(self.buf) - keep
                del self.buf[:len(self.buf) - keep]
                return out
            if k:
                self.skipped_bytes += k
                del self.buf[:k]
            if len(self.buf) < HEADER_BYTES:
                return out
            ftype = self.buf[2]
            ln = self.buf[3] | (self.buf[4] << 8)
            if ftype not in self.accept:
                self.bad_type += 1
                out.append(self._reject(NACK_BAD_TYPE, ftype))
                continue
            if ln != PAYLOAD_LEN[ftype]:
                self.bad_len += 1
                out.append(self._reject(NACK_BAD_LEN, ftype))
                continue
            end = HEADER_BYTES + ln + CRC_BYTES
            if len(self.buf) < end:
                return out
            body = bytes(self.buf[2:HEADER_BYTES + ln])
            got = self.buf[end - 2] | (self.buf[end - 1] << 8)
            if crc16(body) != got:
                self.bad_crc += 1
                out.append(self._reject(NACK_BAD_CRC, ftype))
                continue
            pkt = parse(ftype, body[3:])
            del self.buf[:end]
            self.frames_ok += 1
            out.append(pkt)

    def _reject(self, reason: int, ftype: int) -> FrameError:
        del self.buf[:1]
        self.skipped_bytes += 1
        return FrameError(reason, ftype)
