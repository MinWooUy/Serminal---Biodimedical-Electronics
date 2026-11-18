"""Simulated FPGA (protocol v2, CNN only) and the transports the host talks through.

FpgaSim     : byte-level FPGA model: decodes host frames (PING, BEAT_IN), runs the frozen int8 CNN
              (python/quant/int8.py:int_forward_q) and answers PONG / RESULT / NACK exactly as docs/proto_v2.md says.
              cycles = 0 (the simulator does not model timing). Optional fault injection: busy (NACK 4) and
              silently ignored frames, with a seeded RNG.
InProcessLink : host-side transport to an FpgaSim inside the same process, on a VIRTUAL clock (115200 8N1 byte
              time, configurable compute delay), so timeouts and retransmissions are tested fast and
              deterministically. Optional line faults in both directions: flip one bit, drop one byte (+ garbage),
              lose the whole frame.
SerialTransport : the same interface on a real serial port (python/link/serial_link.py).
"""

from dataclasses import dataclass, field

import numpy as np

from python.link import proto_v2 as pv2
from python.link.line_faults import BYTE_TIME_S, LineFaults  # noqa: F401  (re-exported)
from python.quant.int8 import QuantModel, int_forward_q



class FpgaSim:
    def __init__(self, qm: QuantModel, weights_crc32: int, build_id: int = 0x26102600, busy_rate: float = 0.0,
                 ignore_rate: float = 0.0, seed: int = 0):
        self.qm, self.weights_crc32, self.build_id = qm, int(weights_crc32), int(build_id)
        self.busy_rate, self.ignore_rate = busy_rate, ignore_rate
        self.rng = np.random.default_rng(seed)
        self.dec = pv2.Decoder(accept=pv2.HOST_TO_FPGA)
        self.err_count = 0                    # frame errors since reset (NACK err_count)
        self.stats = {"ping": 0, "beats": 0, "nack_crc": 0, "nack_len": 0, "nack_type": 0, "nack_busy": 0,
                      "ignored": 0}

    def reset(self) -> None:
        self.dec = pv2.Decoder(accept=pv2.HOST_TO_FPGA)
        self.err_count = 0

    def feed(self, data: bytes) -> list[bytes]:
        """Bytes from the host -> list of reply frames (one per accepted / rejected frame)."""
        out = []
        for ev in self.dec.feed(data):
            if isinstance(ev, pv2.FrameError):
                self.err_count = (self.err_count + 1) & 0xFFFF
                self.stats[{1: "nack_crc", 2: "nack_len", 3: "nack_type"}[ev.reason]] += 1
                out.append(pv2.encode(pv2.Nack(ev.reason, ev.ftype, self.err_count)))
                continue
            if self.ignore_rate and self.rng.random() < self.ignore_rate:
                self.stats["ignored"] += 1
                continue
            if isinstance(ev, pv2.Ping):
                self.stats["ping"] += 1
                out.append(pv2.encode(pv2.Pong(pv2.PROTO_VER, pv2.MODE_CNN_ONLY, self.build_id, self.weights_crc32)))
            elif isinstance(ev, pv2.BeatIn):
                if self.busy_rate and self.rng.random() < self.busy_rate:
                    self.stats["nack_busy"] += 1
                    out.append(pv2.encode(pv2.Nack(pv2.NACK_BUSY, pv2.T_BEAT_IN, self.err_count)))
                    continue
                self.stats["beats"] += 1
                logits = int_forward_q(self.qm, np.array([ev.z_q]), np.array([ev.rr_q]))[0]
                out.append(pv2.encode(pv2.Result(ev.beat_id, tuple(int(v) for v in logits), int(np.argmax(logits)), 0)))
        return out


# ---------------------------------------------------------------- transports

@dataclass
class InProcessLink:
    """Host-side transport to an in-process FpgaSim on a virtual clock."""
    sim: FpgaSim
    compute_s: float = 0.002                       # virtual FPGA processing time per frame
    down: LineFaults = field(default_factory=LineFaults)       # host -> FPGA
    up: LineFaults = field(default_factory=LineFaults)         # FPGA -> host
    seed: int = 0
    now: float = 0.0
    _rx: list = field(default_factory=list)                    # (arrival time, bytes) towards the host

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        self.faults = {"down": {"lost": 0, "flipped": 0, "dropped": 0}, "up": {"lost": 0, "flipped": 0, "dropped": 0}}

    def write(self, data: bytes) -> None:
        data = self.down.apply(data, self.rng, self.faults["down"])
        self.now += len(data) * BYTE_TIME_S
        t = self.now + self.compute_s
        for fr in self.sim.feed(data):
            fr = self.up.apply(fr, self.rng, self.faults["up"])
            t += len(fr) * BYTE_TIME_S
            if fr:
                self._rx.append((t, fr))

    def read(self, timeout: float) -> bytes:
        """Bytes that arrive within `timeout` (advances the virtual clock); b'' on timeout."""
        deadline = self.now + timeout
        if self._rx and self._rx[0][0] <= deadline:
            t, fr = self._rx.pop(0)
            self.now = max(self.now, t)
            return fr
        self.now = deadline
        return b""

    def clock(self) -> float:
        return self.now
