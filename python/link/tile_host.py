"""Host side of the tile protocol (docs/proto_tile.md): stop-and-wait client + the frozen Lite CNN on real or simulated tiles.

TileHost(transport) talks to any transport with write(bytes) / read(timeout) -> bytes / clock(): SerialTransport (COM port),
TileLink (in-process simulator) or RawLog around either. It never resends anything by itself (the FPGA team's rule, and a
RUN_TILE must never be repeated blindly); any NACK, timeout, wrong SEQ or corrupted answer raises TileLinkError with a
plain-language Vietnamese diagnosis. TileHost.apply(cmd) has the TileModel interface, so tile_sched.run_beat drives the
board exactly as it drives the model.

    smoke(host)                -> the 3x3 example of the FPGA team's README (no golden needed)
    run_golden(host, qm, g, …) -> golden beats through the board, every layer compared bit for bit
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

try:
    from python.link import proto_tile as pt
    from python.link import tile_sched as ts
except ImportError:                       # stand-alone copy in the hand-off package
    import proto_tile as pt
    import tile_sched as ts

LAYERS = ("conv0", "pool0", "conv1", "pool1", "conv2", "pool2", "conv3", "gap", "fc1", "logits")


class TileLinkError(RuntimeError):
    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind                   # "nack", "timeout", "seq", "corrupt", "format"


@dataclass
class HostStats:
    transactions: int = 0
    tx_bytes: int = 0
    rx_bytes: int = 0
    by_cmd: dict = field(default_factory=dict)
    noise: int = 0                         # bytes before A5 5A that were skipped


def _diagnose_nack(cmd: int, code: int) -> str:
    name = pt.NAMES.get(cmd, f"0x{cmd:02X}")
    hint = {
        pt.NACK_CRC: "FPGA nhận khung hỏng (nhiễu dây, sai baud, GND chung?) hoặc LEN > 256.",
        pt.NACK_LEN: "độ dài / cấu hình / địa chỉ không hợp lệ với bitstream (kiểm ROWS = COLS = 7, MEM_DEPTH = 1024, "
                     "CONFIG_POST 8 byte).",
        pt.NACK_NOT_READY: "RUN khi chưa cấu hình, thiếu bias / CONFIG_POST, hoặc bank thiếu byte "
                           "(FPGA vừa reset giữa chừng?).",
        pt.NACK_CMD: "FPGA không biết lệnh này (bitstream cũ?).",
    }.get(code, "mã lạ")
    return f"NACK {code} cho {name}: {pt.NACK_TEXT.get(code, '?')} — {hint}"


class TileHost:
    def __init__(self, transport, timeout: float = 1.0):
        self.t, self.timeout = transport, timeout
        self.seq = 0
        self.cfg: ts.ConfigTile | None = None
        self.dec = pt.Decoder()
        self.stats = HostStats()

    # ------------------------------------------------------------ one request / one answer
    def exchange(self, cmd):
        code, body = pt.encode_cmd(cmd)
        seq = self.seq
        self.seq = (self.seq + 1) & 0xFF
        fr = pt.frame(code, seq, body)
        self.t.write(fr)
        st = self.stats
        st.transactions += 1
        st.tx_bytes += len(fr)
        st.by_cmd[pt.NAMES[code]] = st.by_cmd.get(pt.NAMES[code], 0) + 1
        deadline = self.t.clock() + self.timeout
        pending: list = []
        while True:
            while pending:
                ev = pending.pop(0)
                if isinstance(ev, pt.BadFrame):
                    if ev.why == "bytes before A5 5A":
                        st.noise += len(ev.data)
                        continue
                    self._drop_input()
                    raise TileLinkError("corrupt", f"trả lời hỏng cho {pt.NAMES[code]} (SEQ {seq}): {ev.why}. "
                                                   "Nhiễu dây / sai baud? Không gửi lại.")
                if ev.seq != seq:
                    self._drop_input()
                    raise TileLinkError("seq", f"trả lời SEQ {ev.seq} cho lệnh SEQ {seq} ({pt.NAMES[code]}): "
                                               "lệch nhịp với FPGA (trả lời trễ của lệnh trước?).")
                if isinstance(ev, pt.Nack):
                    raise TileLinkError("nack", _diagnose_nack(ev.cmd, ev.code))
                want = pt.RESULT if code == pt.RUN_TILE else pt.ACK
                got = pt.RESULT if isinstance(ev, pt.Result) else pt.ACK
                if got != want or (isinstance(ev, pt.Ack) and ev.cmd != code):
                    raise TileLinkError("format", f"chờ {pt.NAMES[want]} cho {pt.NAMES[code]}, nhận {ev}")
                return ev
            left = deadline - self.t.clock()
            if left <= 0:
                what = "RESULT" if code == pt.RUN_TILE else "ACK"
                raise TileLinkError("timeout", f"không có {what} cho {pt.NAMES[code]} (SEQ {seq}) sau {self.timeout} s. "
                                               "Kiểm dây TX/RX, GND, cổng COM, bitstream đã nạp, nút reset (KEY0). "
                                               "Không gửi lại (RUN không được lặp mù).")
            data = self.t.read(left)
            if data:
                st.rx_bytes += len(data)
                pending += self.dec.feed(data)

    def _drop_input(self, quiet: float = 0.25) -> None:
        """Discard whatever is still arriving until the line has been silent for `quiet` seconds."""
        while self.t.read(quiet):
            pass
        self.dec = pt.Decoder()

    def resync(self) -> None:
        """After an error: wait out the FPGA's 100 ms receive timeout and any late answer, forget partial input."""
        self._drop_input()
        self.cfg = None

    # ------------------------------------------------------------ TileModel interface for tile_sched.run_beat
    def apply(self, cmd):
        ev = self.exchange(cmd)
        if isinstance(cmd, ts.ConfigTile):
            self.cfg = cmd
        if not isinstance(cmd, ts.RunTile):
            return None
        if ev.post is None:
            raise TileLinkError("format", "bitstream trả RESULT chỉ có raw (POST_ENABLE = 0): nạp lại bản POST_ENABLE = 1.")
        return np.array(ev.raw).reshape(ts.ROWS, ts.COLS), np.array(ev.post).reshape(ts.ROWS, ts.COLS)

    def status(self) -> dict:
        ev = self.exchange(pt.GetStatus())
        f = ev.flags or 0
        return {"flags": f, "configured": bool(f & 1), "ifmap_staging_dirty": bool(f & 2),
                "weight_staging_dirty": bool(f & 4), "bias_preload": bool(f & 8), "post_configured": bool(f & 16)}


# ---------------------------------------------------------------- checks
SMOKE_X = (1, -2, 3, 4, -1)
SMOKE_W = (1, 0, -1, 2, 1, 0, 0, 1, 2)
SMOKE_RAW = [[-2, 0, 4], [-6, -1, 11], [4, 10, 2]]


def smoke(host: TileHost) -> dict:
    """README example: X = [1,-2,3,4,-1], filters [1,0,-1], [2,1,0], [0,1,2] -> 3x3 raw results; post with shift 0, no ReLU."""
    for c in (ts.WriteIfmap(0, SMOKE_X), ts.WriteWeight(0, SMOKE_W), ts.ConfigPost((0,) * ts.COLS, False),
              ts.ConfigTile(3, 1, 3, 3, 0, 0, False)):
        host.apply(c)
    raw, post = host.apply(ts.RunTile("smoke", 0, 0))
    got = raw[:3, :3].tolist()
    return {"ok": got == SMOKE_RAW and post[:3, :3].tolist() == SMOKE_RAW, "raw": got, "expected": SMOKE_RAW}


def run_golden(host: TileHost, qm, g: dict, beats, log=print, upto: str | None = None) -> dict:
    """Golden beats through `host`; compares every layer (up to `upto`) bit for bit. A failed beat is reported, the link
    re-synchronised and the planner state forgotten (everything is re-sent for the next beat); nothing is retried."""
    names = LAYERS if upto is None else LAYERS[:LAYERS.index(upto) + 1]
    names = tuple(k for k in names if k in g)              # the hand-off golden has every layer only for some beats
    res = {"beats": 0, "exact": 0, "mismatch": [], "failed": [], "pred_ok": 0, "layer_mismatch": {k: 0 for k in names}}
    state: dict = {}
    t0 = host.t.clock()
    for n, i in enumerate(beats):
        res["beats"] += 1
        try:
            r = ts.run_beat(qm, g["z_q"][i], g["rr_q"][i], host, state, upto=upto)
        except TileLinkError as e:
            res["failed"].append({"beat": int(i), "kind": e.kind, "error": str(e)})
            log(f"nhịp {i}: LỖI ({e.kind}) {e}")
            host.resync()
            state.clear()
            continue
        bad = [k for k in names if not np.array_equal(r.layers[k], g[k][i])]
        for k in bad:
            res["layer_mismatch"][k] += 1
        if upto is None and "pred" in g:
            res["pred_ok"] += int(r.pred == g["pred"][i])
        if bad:
            res["mismatch"].append({"beat": int(i), "layers": bad})
            log(f"nhịp {i}: SAI ở {bad}")
        else:
            res["exact"] += 1
        if (n + 1) % 50 == 0:
            log(f"  {n + 1} nhịp, {res['exact']} trùng từng bit, {len(res['failed'])} lỗi đường truyền")
    res["seconds"] = host.t.clock() - t0
    res["transactions"] = host.stats.transactions
    res["bytes"] = host.stats.tx_bytes + host.stats.rx_bytes
    return res
