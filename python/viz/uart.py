"""UART framing between the device (or a simulated device) and the host monitor. Spec: docs/uart_protocol.md.

Frame: 0xA5 0x5A | type u8 | len u8 | payload (len bytes) | crc u16 (little-endian),
crc = CRC-16/CCITT-FALSE over type, len, payload (poly 0x1021, init 0xFFFF, no reflection, xorout 0;
check("123456789") = 0x29B1). All multi-byte fields little-endian.
  0x01 SAMPLES  first_idx u32, n u8 (1..16), adc i16 x n, filt i32 x n           len 5 + 6n
  0x02 BEAT     r_idx u32, emit_idx u32, pred u8, flags u8, logits i32 x 4,
                rr_prev u16, rr_next u16, rr_sum8 u16, rr_n8 u8                  len 33
  0x03 STATS    sample_idx u32, n_beats u32, cycles_last u32, cycles_max u32, cpu_hz u32,
                adc_gain u16, filt_frac u8, logit_frac u8, beat_digest u32       len 28
beat_digest = CRC-32/IEEE (zlib.crc32; check("123456789") = 0xCBF43926) accumulated over the payloads of all
BEAT frames since the start of the run, in order.
The decoder is a byte-stream state machine that re-synchronises on the next 0xA5 0x5A after any
bad header (unknown type / invalid length), CRC error or failed host check (HostDecoder).
"""

import struct
import zlib
from dataclasses import dataclass, field

SYNC = b"\xa5\x5a"
T_SAMPLES, T_BEAT, T_STATS = 0x01, 0x02, 0x03
MAX_SAMPLES_PER_FRAME = 16
FLAG_SEARCHBACK = 0x01
FLAG_REFINE_EDGE = 0x02          # integer refinement hit the window edge -> coarse R kept
CRC_BYTES = 2
OVERHEAD = len(SYNC) + 2 + CRC_BYTES          # sync, type, len, crc

_BEAT = struct.Struct("<IIBB4iHHHB")
_STATS = struct.Struct("<IIIIIHBBI")
LEN_BEAT, LEN_STATS = _BEAT.size, _STATS.size


def _crc16_table(poly: int = 0x1021) -> tuple:
    t = []
    for b in range(256):
        c = b << 8
        for _ in range(8):
            c = ((c << 1) ^ poly) & 0xFFFF if c & 0x8000 else (c << 1) & 0xFFFF
        t.append(c)
    return tuple(t)


CRC16_TABLE = _crc16_table()


def crc16(data: bytes, crc: int = 0xFFFF) -> int:
    """CRC-16/CCITT-FALSE: poly 0x1021, init 0xFFFF, MSB first (no reflection), no final XOR."""
    for b in data:
        crc = ((crc << 8) & 0xFFFF) ^ CRC16_TABLE[(crc >> 8) ^ b]
    return crc


def crc32_update(digest: int, data: bytes) -> int:
    """CRC-32/IEEE (as zlib.crc32), chained: crc32_update(crc32_update(0, a), b) == crc32(a + b)."""
    return zlib.crc32(data, digest) & 0xFFFFFFFF


@dataclass(frozen=True)
class Samples:
    first_idx: int
    adc: tuple
    filt: tuple


@dataclass(frozen=True)
class Beat:
    r_idx: int
    emit_idx: int
    pred: int
    flags: int
    logits: tuple
    rr_prev: int
    rr_next: int
    rr_sum8: int
    rr_n8: int


@dataclass(frozen=True)
class Stats:
    sample_idx: int
    n_beats: int
    cycles_last: int
    cycles_max: int
    cpu_hz: int                  # 0 = unknown (PC simulation): cycle fields are not meaningful
    adc_gain: int
    filt_frac: int
    logit_frac: int
    beat_digest: int             # CRC-32 over the payloads of all BEAT frames sent so far


def frame(ftype: int, payload: bytes) -> bytes:
    if len(payload) > 255:
        raise ValueError("payload longer than 255 bytes")
    body = bytes((ftype, len(payload))) + payload
    return SYNC + body + struct.pack("<H", crc16(body))


def encode_samples(first_idx: int, adc, filt) -> bytes:
    n = len(adc)
    if not 1 <= n <= MAX_SAMPLES_PER_FRAME or len(filt) != n:
        raise ValueError(f"1..{MAX_SAMPLES_PER_FRAME} samples per frame")
    return frame(T_SAMPLES, struct.pack(f"<IB{n}h{n}i", first_idx, n, *adc, *filt))


def beat_payload(b: Beat) -> bytes:
    return _BEAT.pack(b.r_idx, b.emit_idx, b.pred, b.flags, *b.logits, b.rr_prev, b.rr_next, b.rr_sum8, b.rr_n8)


def encode_beat(b: Beat) -> bytes:
    return frame(T_BEAT, beat_payload(b))


def encode_stats(s: Stats) -> bytes:
    return frame(T_STATS, _STATS.pack(s.sample_idx, s.n_beats, s.cycles_last, s.cycles_max, s.cpu_hz, s.adc_gain,
                                      s.filt_frac, s.logit_frac, s.beat_digest))


def beat_digest(beats) -> int:
    d = 0
    for b in beats:
        d = crc32_update(d, beat_payload(b))
    return d


def _valid_len(ftype: int, ln: int) -> bool:
    if ftype == T_SAMPLES:
        return ln >= 11 and (ln - 5) % 6 == 0 and (ln - 5) // 6 <= MAX_SAMPLES_PER_FRAME
    return (ftype, ln) in ((T_BEAT, LEN_BEAT), (T_STATS, LEN_STATS))


def _parse(ftype: int, p: bytes):
    if ftype == T_SAMPLES:
        first, n = struct.unpack_from("<IB", p)
        if len(p) != 5 + 6 * n:
            return None
        v = struct.unpack_from(f"<{n}h{n}i", p, 5)
        return Samples(first, v[:n], v[n:])
    if ftype == T_BEAT:
        v = _BEAT.unpack(p)
        return Beat(v[0], v[1], v[2], v[3], v[4:8], v[8], v[9], v[10], v[11])
    return Stats(*_STATS.unpack(p))


@dataclass
class FrameDecoder:
    """Feed arbitrary byte chunks; returns the decoded packets in order; call end() when the stream ends.

    Acceptance: valid header and CRC-16 match, then validate() (HostDecoder). A frame that passes the CRC
    but fails validate() is rejected and counted in transport_error. Every rejection discards ONE byte and
    rescans for the sync word."""
    buf: bytearray = field(default_factory=bytearray)
    frames_ok: int = 0
    bad_header: int = 0
    bad_crc: int = 0
    transport_error: int = 0
    skipped_bytes: int = 0
    ended: bool = False

    def feed(self, data: bytes) -> list:
        self.buf += data
        return self._run()

    def end(self) -> list:
        """End of stream: leftover bytes (an incomplete frame) are counted as skipped."""
        self.ended = True
        out = self._run()
        self.skipped_bytes += len(self.buf)
        self.buf.clear()
        return out

    def validate(self, pkt) -> bool:            # host-side plausibility checks (HostDecoder)
        return True

    def _run(self) -> list:
        out = []
        while True:
            k = self.buf.find(SYNC)
            if k < 0:
                keep = 1 if self.buf[-1:] == SYNC[:1] and not self.ended else 0   # a split sync word may continue
                self.skipped_bytes += len(self.buf) - keep
                del self.buf[:len(self.buf) - keep]
                return out
            if k:
                self.skipped_bytes += k
                del self.buf[:k]
            if len(self.buf) < 4:
                if self.ended:
                    self._drop_one()
                    continue
                return out
            ftype, ln = self.buf[2], self.buf[3]
            if not _valid_len(ftype, ln):
                self.bad_header += 1
                self._drop_one()
                continue
            end = 4 + ln + CRC_BYTES
            if len(self.buf) < end:
                if self.ended:
                    self._drop_one()
                    continue
                return out
            ok = crc16(self.buf[2:4 + ln]) == struct.unpack_from("<H", self.buf, 4 + ln)[0]
            pkt = _parse(ftype, bytes(self.buf[4:4 + ln])) if ok else None
            if pkt is None:
                self.bad_crc += 1
                self._drop_one()
                continue
            if not self.validate(pkt):
                self.transport_error += 1
                self._drop_one()
                continue
            del self.buf[:end]
            self.frames_ok += 1
            out.append(pkt)

    def _drop_one(self):
        del self.buf[:1]
        self.skipped_bytes += 1


@dataclass
class HostDecoder(FrameDecoder):
    """FrameDecoder + host plausibility checks (no protocol change):
      SAMPLES: always accepted. A discontinuous first_idx is recorded (a forward gap adds to samples_lost,
               every discontinuity counts in resyncs) and the host re-synchronises to the new first_idx
               (a backward jump, e.g. a device reset, also restarts the received range);
      BEAT   : rejected unless pred in 0..3 and r_idx inside the received sample range [start_idx, next_idx)."""
    start_idx: int | None = None
    next_idx: int | None = None
    samples_lost: int = 0
    resyncs: int = 0

    def validate(self, pkt) -> bool:
        if isinstance(pkt, Samples):
            end = pkt.first_idx + len(pkt.adc)
            if self.next_idx is None:
                self.start_idx = pkt.first_idx
            elif pkt.first_idx != self.next_idx:
                self.resyncs += 1
                if pkt.first_idx > self.next_idx:
                    self.samples_lost += pkt.first_idx - self.next_idx
                else:
                    self.start_idx = pkt.first_idx
            self.next_idx = end
            return True
        if isinstance(pkt, Beat):
            return 0 <= pkt.pred <= 3 and self.next_idx is not None and self.start_idx <= pkt.r_idx < self.next_idx
        return True
