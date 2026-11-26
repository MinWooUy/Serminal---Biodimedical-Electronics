"""Simulated layer accelerator at byte level (layer_cmd_controller.v + packet_rx.v; RTL Serminal dev ca7562e) and its transport.

LayerFpgaSim : packet_rx.v (A5 5A sync, LEN > MAX_PAYLOAD -> NACK 1 right after the header, CRC -> NACK 1, 100 ms inter-byte
               timeout drops a partial packet silently) + layer_cmd_controller.v (per-command length checks, NACK 2 / 3 / 4,
               ACK, RESULT) on top of sched.LayerModel (banks, flags, layer arithmetic). Bytes that arrive while the FPGA
               handles a command or transmits its answer are DROPPED, as in the RTL (the receiver is enabled only in IDLE).
LayerLink    : host-side transport on a VIRTUAL clock (the board's baud, 8N1); same write / read / clock interface as
               serial_link.SerialTransport, with the same optional line faults (LineFaults).

Not modelled: the real cycle count of a layer (LayerModel gives an estimate), UART bit timing inside a byte.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

import numpy as np

from python.link.line_faults import LineFaults
from python.link.proto_v2 import crc16

from . import codec as pl
from . import sched as ls

RX_TIMEOUT_S = 0.1                     # RX_TIMEOUT_CLKS = CLK_FREQ / 10


class LayerFpgaSim:
    def __init__(self, hw: ls.HwConfig | None = None, model: ls.LayerModel | None = None, strict: bool = False):
        self.hw = hw or (model.hw if model else ls.HwConfig())
        self.model = model or ls.LayerModel(self.hw, strict=strict)
        self.byte_time = 10 / self.hw.baud
        self.stats = {"frames": 0, "acks": 0, "results": 0, "nack": {1: 0, 2: 0, 3: 0, 4: 0}, "dropped_busy": 0,
                      "timeouts": 0, "by_cmd": {}}
        self.reset()

    def reset(self) -> None:
        """Reset button: clears configuration, banks' flags, GAP memory and the receiver."""
        self.model.reset()
        self.state, self.busy_until, self.last_t = "SOF0", -1.0, None
        self.cmd = self.seq = self.length = 0
        self.body, self.crc_lo = bytearray(), 0

    # ------------------------------------------------------------ receiver (packet_rx.v)
    def feed(self, data: bytes, t0: float) -> list[tuple[float, bytes]]:
        """Bytes whose last bit ends at t0 + (i + 1) * byte_time -> [(time the answer is fully sent, answer frame)]."""
        out = []
        bt = self.byte_time
        i, n = 0, len(data)
        while i < n:
            t = t0 + (i + 1) * bt
            if t <= self.busy_until:
                self.state = "SOF0"
                self.stats["dropped_busy"] += 1
                i += 1
                continue
            if self.state != "SOF0" and self.last_t is not None and t - self.last_t > RX_TIMEOUT_S:
                self.state = "SOF0"
                self.stats["timeouts"] += 1
            if self.state == "BODY":                              # bulk path: a body cannot be interrupted by a busy window
                m = min(n - i, self.length - len(self.body))
                self.body += data[i:i + m]
                self.last_t = t0 + (i + m) * bt
                i += m
                if len(self.body) == self.length:
                    self.state = "CRC0"
                continue
            self.last_t = t
            ans = self._byte(data[i])
            i += 1
            if ans is not None:
                cmd_s, fr = ans
                done = t + cmd_s + len(fr) * bt
                self.busy_until = done
                out.append((done, fr))
        return out

    def _byte(self, b: int):
        s = self.state
        if s == "SOF0":
            self.state = "SOF1" if b == 0xA5 else "SOF0"
        elif s == "SOF1":
            self.state = "CMD" if b == 0x5A else ("SOF1" if b == 0xA5 else "SOF0")
        elif s == "CMD":
            self.cmd, self.state = b, "SEQ"
        elif s == "SEQ":
            self.seq, self.state = b, "LEN0"
        elif s == "LEN0":
            self.length, self.state = b, "LEN1"
        elif s == "LEN1":
            self.length |= b << 8
            self.body = bytearray()
            if self.length > self.hw.max_payload:
                self.state = "SOF0"
                return self._nack(self.cmd, pl.NACK_CRC)
            self.state = "BODY" if self.length else "CRC0"
        elif s == "CRC0":
            self.crc_lo, self.state = b, "CRC1"
        elif s == "CRC1":
            self.state = "SOF0"
            head = struct.pack("<BBH", self.cmd, self.seq, self.length)
            if (b << 8 | self.crc_lo) != crc16(head + bytes(self.body)):
                return self._nack(self.cmd, pl.NACK_CRC)
            return self._command(self.cmd, bytes(self.body))
        return None

    # ------------------------------------------------------------ controller (layer_cmd_controller.v)
    def _answer(self, rsp: int, body: bytes, cmd_s: float = 0.0):
        if len(body) > self.hw.max_payload:
            raise ls.LayerUndefined(f"answer of {len(body)} B does not fit the {self.hw.max_payload} B transmit payload")
        return cmd_s, pl.frame(rsp, self.seq, body)

    def _nack(self, cmd: int, code: int):
        self.stats["nack"][code] += 1
        return self._answer(pl.NACK, bytes([cmd, code]))

    def _ack(self, cmd: int, cmd_s: float = 0.0):
        self.stats["acks"] += 1
        return self._answer(pl.ACK, bytes([cmd, 0]), cmd_s)

    def _result(self, body: bytes, cmd_s: float = 0.0):
        self.stats["results"] += 1
        return self._answer(pl.RESULT, body, cmd_s)

    def _command(self, code: int, p: bytes):
        self.stats["frames"] += 1
        name = pl.NAMES.get(code, f"0x{code:02X}")
        self.stats["by_cmd"][name] = self.stats["by_cmd"].get(name, 0) + 1
        clk = self.hw.clk_hz
        if code not in pl.NAMES or code in (pl.CONFIG_ALL, pl.LOAD_COMMIT, pl.RUN_ALL, pl.GET_RESULT) or code >= 0x80:
            return self._nack(code, pl.NACK_CMD)                              # reserved CMDs 0x11..0x14 have no FSM branch yet
        try:
            cmd = pl.decode_cmd(code, p)
        except ValueError:
            return self._nack(code, pl.NACK_STATE if code == pl.READ_FMAP else pl.NACK_ARG)   # the RTL's READ_FMAP quirk
        try:
            res = self.model.apply(cmd)
        except ls.LayerNack as e:
            return self._nack(code, e.code)
        if code == pl.RUN:
            status, cycles = res
            return self._result(struct.pack("<BI", status, cycles), cycles / clk)
        if code == pl.READ_GAP:
            return self._result(struct.pack(f"<{len(res)}H", *res), len(res) / clk)
        if code == pl.READ_FMAP:
            return self._result(res, 2 * len(res) / clk)
        if code == pl.INFO:
            return self._result(res)
        return self._ack(code, (len(p) - 2) / clk if code in (pl.WRITE_X, pl.WRITE_W, pl.WRITE_SHIFT) else 0.0)


@dataclass
class LayerLink:
    """Host-side transport to an in-process LayerFpgaSim on a virtual clock."""
    sim: LayerFpgaSim
    down: LineFaults = field(default_factory=LineFaults)       # host -> FPGA
    up: LineFaults = field(default_factory=LineFaults)         # FPGA -> host
    seed: int = 0
    now: float = 0.0
    _rx: list = field(default_factory=list)                    # (time fully received by the host, bytes)

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        self.faults = {"down": {"lost": 0, "flipped": 0, "dropped": 0}, "up": {"lost": 0, "flipped": 0, "dropped": 0}}

    def write(self, data: bytes) -> None:
        data = self.down.apply(data, self.rng, self.faults["down"])
        for t, fr in self.sim.feed(data, self.now):
            fr = self.up.apply(fr, self.rng, self.faults["up"])
            if fr:
                self._rx.append((t, fr))
        self._rx.sort(key=lambda x: x[0])
        self.now += len(data) * self.sim.byte_time

    def read(self, timeout: float) -> bytes:
        """Bytes fully arrived within `timeout` (advances the virtual clock); b'' on timeout."""
        deadline = self.now + timeout
        if self._rx and self._rx[0][0] <= deadline:
            t, fr = self._rx.pop(0)
            self.now = max(self.now, t)
            return fr
        self.now = deadline
        return b""

    def wait(self, seconds: float) -> None:
        self.now += seconds

    def clock(self) -> float:
        return self.now
