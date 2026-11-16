"""Simulated tile accelerator at byte level (docs/proto_tile.md; RTL Serminal dev 4ad6c8b) and its in-process transport.

TileFpgaSim : packet_rx.v (A5 5A sync, LEN > 256 -> NACK 1 right after the header, CRC -> NACK 1, 100 ms inter-byte
              timeout drops a partial packet silently) + host_command_controller.v (per-command length / config checks,
              NACK 2 / 3 / 4, ACK, RESULT 196 + 49 B) on top of tile_sched.TileModel (banks, validity, tile arithmetic).
              Bytes that arrive while the FPGA handles a command or transmits its answer are DROPPED, as in the RTL
              (the receiver is enabled only in IDLE).
TileLink    : host-side transport on a VIRTUAL clock (115200 8N1 byte time); same write / read / clock interface as
              python/link/fpga_sim.InProcessLink, with the same optional line faults (LineFaults).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

import numpy as np

try:
    from python.link import proto_tile as pt
    from python.link import tile_sched as ts
    from python.link.line_faults import BYTE_TIME_S, LineFaults
    from python.link.proto_v2 import crc16
except ImportError:                       # stand-alone copy in the hand-off package
    import proto_tile as pt
    import tile_sched as ts
    from line_faults import BYTE_TIME_S, LineFaults
    from proto_v2 import crc16

CLK_HZ = 50_000_000
RX_TIMEOUT_S = 0.1                     # RX_TIMEOUT_CLKS = CLK_FREQ / 10


class TileFpgaSim:
    def __init__(self, run_s: float = 0.0005, model: ts.TileModel | None = None):
        self.model = model or ts.TileModel()
        self.run_s = run_s                          # virtual compute time of one RUN_TILE (RTL timing not modelled)
        self.stats = {"frames": 0, "acks": 0, "results": 0, "nack": {1: 0, 2: 0, 3: 0, 4: 0}, "dropped_busy": 0,
                      "timeouts": 0, "by_cmd": {}}
        self.reset()

    def reset(self) -> None:
        """KEY0: clears configuration, validity maps, bias flags and the receiver."""
        self.model.reset()
        self.state, self.busy_until, self.last_t = "SOF0", -1.0, None
        self.cmd = self.seq = self.length = 0
        self.body, self.crc_lo = bytearray(), 0

    # ------------------------------------------------------------ receiver (packet_rx.v)
    def feed(self, data: bytes, t0: float) -> list[tuple[float, bytes]]:
        """Bytes whose last bit ends at t0 + (i + 1) * BYTE_TIME -> [(time the answer is fully sent, answer frame)]."""
        out = []
        for i, b in enumerate(data):
            t = t0 + (i + 1) * BYTE_TIME_S
            if t <= self.busy_until:
                self.state = "SOF0"
                self.stats["dropped_busy"] += 1
                continue
            if self.state != "SOF0" and self.last_t is not None and t - self.last_t > RX_TIMEOUT_S:
                self.state = "SOF0"
                self.stats["timeouts"] += 1
            self.last_t = t
            ans = self._byte(b)
            if ans is not None:
                cmd_s, fr = ans
                done = t + cmd_s + len(fr) * BYTE_TIME_S
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
            if self.length > ts.MAX_PAYLOAD:
                self.state = "SOF0"
                return self._nack(self.cmd, pt.NACK_CRC)
            self.state = "BODY" if self.length else "CRC0"
        elif s == "BODY":
            self.body.append(b)
            if len(self.body) == self.length:
                self.state = "CRC0"
        elif s == "CRC0":
            self.crc_lo, self.state = b, "CRC1"
        elif s == "CRC1":
            self.state = "SOF0"
            head = struct.pack("<BBH", self.cmd, self.seq, self.length)
            if (b << 8 | self.crc_lo) != crc16(head + bytes(self.body)):
                return self._nack(self.cmd, pt.NACK_CRC)
            return self._command(self.cmd, bytes(self.body))
        return None

    # ------------------------------------------------------------ controller (host_command_controller.v)
    def _answer(self, rsp: int, body: bytes, cmd_s: float = 0.0):
        return cmd_s, pt.frame(rsp, self.seq, body)

    def _nack(self, cmd: int, code: int):
        self.stats["nack"][code] += 1
        return self._answer(pt.NACK, bytes([cmd, code]))

    def _ack(self, cmd: int, extra: bytes = b"", cmd_s: float = 0.0):
        self.stats["acks"] += 1
        return self._answer(pt.ACK, bytes([cmd, 0]) + extra, cmd_s)

    def _command(self, cmd: int, p: bytes):
        self.stats["frames"] += 1
        name = pt.NAMES.get(cmd, f"0x{cmd:02X}")
        self.stats["by_cmd"][name] = self.stats["by_cmd"].get(name, 0) + 1
        n = len(p)
        try:
            if cmd in (pt.WRITE_IFMAP, pt.WRITE_WEIGHT):
                if n < 3 or struct.unpack("<H", p[:2])[0] + n - 2 > ts.MEM_DEPTH:
                    return self._nack(cmd, pt.NACK_LEN)
                vals = tuple(v - 256 if v > 127 else v for v in p[2:])
                cls = ts.WriteIfmap if cmd == pt.WRITE_IFMAP else ts.WriteWeight
                self.model.apply(cls(struct.unpack("<H", p[:2])[0], vals))
                return self._ack(cmd, cmd_s=(n - 2) / CLK_HZ)
            if cmd == pt.CONFIG_TILE:
                if n not in (10, 11) or (n == 11 and p[10] > 1):
                    return self._nack(cmd, pt.NACK_LEN)
                k, step, rows, cols, xb, wb = struct.unpack("<HHBBHH", p[:10])
                self.model.apply(ts.ConfigTile(k, step, rows, cols, xb, wb, n == 11 and p[10] == 1))
                return self._ack(cmd)
            if cmd == pt.WRITE_BIAS:
                if n < 6 or n % 4 != 2 or struct.unpack("<H", p[:2])[0] + (n - 2) // 4 > ts.COLS:
                    return self._nack(cmd, pt.NACK_LEN)
                self.model.apply(ts.WriteBias(struct.unpack("<H", p[:2])[0], struct.unpack(f"<{(n - 2) // 4}i", p[2:])))
                return self._ack(cmd)
            if cmd == pt.CONFIG_POST:
                if n != ts.COLS + 1 or p[ts.COLS] > 1:
                    return self._nack(cmd, pt.NACK_LEN)
                self.model.apply(ts.ConfigPost(tuple(v - 256 if v > 127 else v for v in p[:ts.COLS]), p[ts.COLS] == 1))
                return self._ack(cmd)
            if cmd == pt.GET_STATUS:
                if n:
                    return self._nack(cmd, pt.NACK_LEN)
                return self._ack(cmd, bytes([self.model.status()]))
            if cmd == pt.RUN_TILE:
                if n:
                    return self._nack(cmd, pt.NACK_LEN)
                raw, post = self.model.apply(ts.RunTile("sim", 0, 0))
                body = struct.pack(f"<{raw.size}i", *raw.reshape(-1).tolist())
                if ts.POST_ENABLE:
                    body += struct.pack(f"<{post.size}b", *post.reshape(-1).tolist())
                self.stats["results"] += 1
                return self._answer(pt.RESULT, body, self.run_s)
        except ts.TileNack as e:
            return self._nack(cmd, e.code)
        return self._nack(cmd, pt.NACK_CMD)


@dataclass
class TileLink:
    """Host-side transport to an in-process TileFpgaSim on a virtual clock."""
    sim: TileFpgaSim
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
        self.now += len(data) * BYTE_TIME_S

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
