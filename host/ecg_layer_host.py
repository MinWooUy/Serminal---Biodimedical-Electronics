"""Run whole CNN layers on the FPGA (ecg_accel_top) and compare against golden data.

Two modes:
  python ecg_layer_host.py COM4 [--record rec101_93600 --beat 0 --last-layer 3]
        Connect over UART, query the hardware configuration (INFO), run layers
        0..last-layer and compare every layer with the golden output.
  python ecg_layer_host.py --dry-run trace.hex [--last-layer N]
        No board / pyserial needed: write the packet sequence (commands + expected
        replies) to a hex file so a testbench can replay it at packet level.

Everything that depends on the synthesized hardware (PE array size, clock, UART baud,
SRAM depths, limits) is read from the FPGA with the INFO command (0x15) and printed as
reported.  With an older bitstream that does not know INFO, pass --array RxC (and
--clk-mhz) and the report marks those values as "assumed".

Report detail on screen (the log file always receives everything):
  (default)  compact report: hardware, one table per layer, summary
  -v         + per-step data statistics and one line per UART packet
  -vv        + first payload bytes of each packet

Isolating a failure:
  --last-layer N     run only up to layer N
  --first-layer N    load the GOLDEN output of layer N-1 as input and start at layer N
                     (layer N is tested alone, errors of earlier layers cannot leak in)

Layout (matches conv1d_layer_top): input/output are channel-last X[t][c], padded;
weights are [Cout][K] with K = j*Cin + i (weights.h stores [Cout][Cin][Kw], so transpose).
"""
import argparse
import math
import re
import struct
import sys
import textwrap
import time
import zlib
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if not (ROOT / "export" / "weights.h").exists():           # started from another directory
    ROOT = Path.cwd()

# ---- command codes (layer_cmd_controller) ----
CMD_WRITE_X, CMD_WRITE_W, CMD_WRITE_BIAS, CMD_WRITE_SHIFT = 0x08, 0x09, 0x0A, 0x0B
CMD_SWAP, CMD_CONFIG, CMD_RUN, CMD_READ_GAP, CMD_READ_FMAP = 0x0C, 0x0D, 0x0E, 0x0F, 0x10

# CMD pipeline nhieu layer: moi chi ghi nhan hop dong, chua gui trong host nay.
CMD_CONFIG_ALL, CMD_LOAD_COMMIT, CMD_RUN_ALL, CMD_GET_RESULT = 0x11, 0x12, 0x13, 0x14
CMD_INFO = 0x15
RSP_ACK, RSP_NACK, RSP_RESULT = 0x80, 0x81, 0x82
CMD_NAMES = {CMD_WRITE_X: "WRITE_X", CMD_WRITE_W: "WRITE_W", CMD_WRITE_BIAS: "WRITE_BIAS",
             CMD_WRITE_SHIFT: "WRITE_SHIFT", CMD_SWAP: "SWAP", CMD_CONFIG: "CONFIG", CMD_RUN: "RUN",
             CMD_READ_GAP: "READ_GAP", CMD_READ_FMAP: "READ_FMAP", CMD_INFO: "INFO",
             CMD_CONFIG_ALL: "CONFIG_ALL", CMD_LOAD_COMMIT: "LOAD_COMMIT",
             CMD_RUN_ALL: "RUN_ALL", CMD_GET_RESULT: "GET_RESULT"}

RSP_NAMES = {RSP_ACK: "ACK", RSP_NACK: "NACK", RSP_RESULT: "RESULT"}
NACK_CODES = {1: "frame/CRC error (packet corrupted on the wire)",
              2: "bad argument (length, address beyond memory, or invalid configuration)",
              3: "bad state (e.g. RUN before CONFIG, or layer still busy)",
              4: "unknown command (wrong bitstream loaded?)"}

INFO_FMT = "<BBBBIIHHHHH"                         # 22 bytes, see layer_cmd_controller.v
INFO_SIZE = struct.calcsize(INFO_FMT)
FRAME_OVERHEAD = 8                                 # A5 5A cmd seq len(2) ... crc(2)
WIDTH = 78

# ---- CNN Lite model: (Cout, Cin, Kw, pool, gap, pad_next, out_len) ----
LAYERS = [
    dict(cout=16, cin=1,  kw=7, pool=1, gap=0, pad_next=2, out_len=360),
    dict(cout=32, cin=16, kw=5, pool=1, gap=0, pad_next=1, out_len=180),
    dict(cout=64, cin=32, kw=3, pool=1, gap=0, pad_next=1, out_len=90),
    dict(cout=64, cin=64, kw=3, pool=0, gap=1, pad_next=0, out_len=45),
]
POOL_SHAPES = [(16, 180), (32, 90), (64, 45)]            # (C, L) after pool of layer 0..2


def c_array(header: str, ctype: str, name: str) -> np.ndarray:
    m = re.search(rf"static\s+const\s+{ctype}\s+{name}\s*\[[^=]+\]\s*=\s*\{{(.*?)\}};", header, re.S)
    if m is None:
        raise RuntimeError(f"{name} not found in weights.h")
    return np.array([int(v) for v in re.findall(r"-?\d+", m.group(1))], dtype=np.int64)


def load_model():
    header = (ROOT / "export" / "weights.h").read_text(encoding="utf-8")
    model = []
    for n, L in enumerate(LAYERS):
        w = c_array(header, "int8_t", f"CONV{n}_W").reshape(L["cout"], L["cin"], L["kw"])
        w_acc = w.transpose(0, 2, 1).reshape(L["cout"], L["kw"] * L["cin"])    # [o][j][i] -> [Cout][K]
        model.append(dict(L, k=L["kw"] * L["cin"], step=L["cin"], weight=w_acc.reshape(-1),
                          bias=c_array(header, "int32_t", f"CONV{n}_B"),
                          shift=c_array(header, "int8_t", f"CONV{n}_SHIFT")))
    return model


def load_beat(record: str, beat: int):
    gold = ROOT / "export" / "golden"

    def arr(name, dtype, shape):
        return np.fromfile(gold / f"{record}_beat_{name}.bin", dtype=dtype).reshape(shape)[beat]

    z = arr("z_q", np.int8, (-1, 360)).astype(np.int64)
    x0 = np.concatenate([np.zeros(3, np.int64), z, np.zeros(3, np.int64)])    # padding for K=7
    expect = []
    for n, (shape, L) in enumerate(zip(POOL_SHAPES, LAYERS[:3])):
        p = arr(f"pool{n}", np.int8, (-1, *shape)).astype(np.int64).T          # (L, C)
        pad = L["pad_next"]
        expect.append(np.pad(p, ((pad, pad), (0, 0))).reshape(-1))
    gap = arr("gap", np.int32, (-1, 64)).astype(np.int64)
    return x0, expect, gap


# ---------------------------------------------------------------- hardware description
@dataclass
class HwInfo:
    """Hardware configuration. `reported` is True when it came from the FPGA's INFO reply."""
    rows: int
    cols: int
    clk_hz: int
    baud: int
    ifmap_depth: int
    wgt_depth: int
    feeder_depth: int
    cout_max: int
    max_payload: int
    reported: bool = True
    version: int = 1

    @property
    def pes(self):
        return self.rows * self.cols

    @classmethod
    def from_payload(cls, p: bytes):
        if len(p) != INFO_SIZE:
            raise ValueError(f"INFO reply is {len(p)} B, expected {INFO_SIZE} B")
        ver, rows, cols, _, clk, baud, ifm, wgt, feed, cmax, mp = struct.unpack(INFO_FMT, p)
        return cls(rows, cols, clk, baud, ifm, wgt, feed, cmax, mp, True, ver)

    @classmethod
    def assumed(cls, rows, cols, clk_hz, baud):
        # Remaining limits are the ecg_accel_top defaults.
        return cls(rows, cols, clk_hz, baud, 4096, 16384, 1024, 64, 256, False)

    def to_payload(self):
        return struct.pack(INFO_FMT, self.version, self.rows, self.cols, 0, self.clk_hz, self.baud,
                           self.ifmap_depth, self.wgt_depth, self.feeder_depth, self.cout_max, self.max_payload)


def parse_array(text: str):
    m = re.fullmatch(r"\s*(\d+)\s*[xX]\s*(\d+)\s*", text)
    if not m or int(m.group(1)) < 1 or int(m.group(2)) < 1:
        raise argparse.ArgumentTypeError("expected ROWSxCOLS, e.g. 7x7")
    return int(m.group(1)), int(m.group(2))


# ---------------------------------------------------------------- log
class Log:
    """Everything goes to the log file; the screen shows lines with level <= verbose."""

    def __init__(self, path=None, verbose=0):
        self.verbose = verbose
        self.path = path
        self.f = open(path, "w", encoding="utf-8") if path else None

    def out(self, msg="", level=0):
        if self.f:
            self.f.write(msg + "\n")
            self.f.flush()
        if level <= self.verbose:
            print(msg, flush=True)

    def close(self):
        if self.f:
            self.f.close()


# ---------------------------------------------------------------- devices
class SerialDevice:
    """Real UART link, built on Accelerator.exchange from ecg_host.py."""

    def __init__(self, port, baud, timeout):
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import ecg_host as eh
        self.acc = eh.Accelerator(port, baud, timeout, 3, 3)    # array args only matter for the old tile protocol

    def exchange(self, cmd, payload, rsp, expect=None, cmp_len=None):
        # Data comparison is done by LayerRunner (to report the FIRST wrong position); here we only return the payload.
        return bytes(self.acc.exchange(cmd, payload, rsp).payload)

    def close(self):
        self.acc.close()


class DryRunDevice:
    """Only records the packet sequence + expected replies so a testbench can replay them."""

    def __init__(self):
        self.packets = []

    def exchange(self, cmd, payload, rsp, expect=None, cmp_len=None):
        if rsp == RSP_ACK:
            expect = bytes((cmd, 0))
            cmp_len = 2
        elif expect is None:
            expect = b""
            cmp_len = 0
        elif cmp_len is None:
            cmp_len = len(expect)
        self.packets.append((cmd, bytes(payload), rsp, bytes(expect), cmp_len))
        return expect

    def write_trace(self, path):
        out = bytearray(struct.pack("<I", len(self.packets)))
        for cmd, payload, rsp, expect, cmp_len in self.packets:
            out += bytes((cmd,)) + struct.pack("<H", len(payload)) + payload
            out += bytes((rsp,)) + struct.pack("<HH", len(expect), cmp_len) + expect
        Path(path).write_text("\n".join(f"{b:02x}" for b in out) + "\n")
        return len(out)

    def close(self):
        pass


class LinkError(RuntimeError):
    """Protocol/UART-level failure (NACK, timeout, CRC...), with context for diagnosis."""

    def __init__(self, cmd, name, msg, note=""):
        super().__init__(msg)
        self.cmd, self.name, self.msg, self.note = cmd, name, msg, note


def nack_code(msg: str):
    m = re.search(r"original_cmd=(\d+), error=(\d+)", msg)
    return int(m.group(2)) if m else None


class TracedDevice:
    """Wraps a device: counts packets/bytes, times each packet, logs one line per packet."""

    def __init__(self, dev, log):
        self.dev, self.log = dev, log
        self.packets = self.tx_bytes = self.rx_bytes = 0
        self.t0 = time.monotonic()

    def exchange(self, cmd, payload, rsp, expect=None, cmp_len=None, note=""):
        name = CMD_NAMES.get(cmd, f"0x{cmd:02X}")
        seq = self.packets
        t = time.monotonic()
        head = f"      [{t - self.t0:7.3f}s] #{seq:03d} {name:<11}{len(payload):4d} B  {note:<22}"
        try:
            res = self.dev.exchange(cmd, payload, rsp, expect=expect, cmp_len=cmp_len)
        except Exception as e:                       # noqa: BLE001 - re-raised as LinkError with context
            self.log.out(f"{head} -> ERROR: {e}")
            self.packets += 1
            raise LinkError(cmd, name, str(e), note) from e
        dt = time.monotonic() - t
        self.packets += 1
        self.tx_bytes += len(payload)
        self.rx_bytes += len(res)
        extra = ""
        if self.log.verbose >= 2:
            extra = f"  tx[{payload[:8].hex(' ')}{'..' if len(payload) > 8 else ''}] " \
                    f"rx[{bytes(res[:8]).hex(' ')}{'..' if len(res) > 8 else ''}]"
        self.log.out(f"{head} -> {RSP_NAMES.get(rsp, hex(rsp)):<6}{len(res):4d} B {dt * 1e3:7.1f} ms{extra}", 1)
        return res

    def close(self):
        self.dev.close()


def query_hw(dev, log, args):
    """Ask the FPGA to describe itself. Falls back to --array/--clk-mhz for old bitstreams."""
    fb_rows, fb_cols = args.array if args.array else (3, 3)
    fallback = HwInfo.assumed(fb_rows, fb_cols, int(args.clk_mhz * 1e6), args.baud)
    if isinstance(dev, DryRunDevice):
        return fallback, "ASSUMED (dry run, no board)", ["values taken from --array / --clk-mhz"]
    try:
        hw = HwInfo.from_payload(dev.exchange(CMD_INFO, b"", RSP_RESULT))
        notes = []
        if args.array and args.array != (hw.rows, hw.cols):
            notes.append(f"--array {args.array[0]}x{args.array[1]} ignored: the FPGA reports {hw.rows}x{hw.cols}")
        return hw, "reported by FPGA (INFO command)", notes
    except Exception as e:                           # noqa: BLE001
        if nack_code(str(e)) == 4:
            why = "FPGA answered INFO with 'unknown command': bitstream predates INFO"
        else:
            why = f"INFO failed: {e}"
        arr = (f"PE array {fb_rows}x{fb_cols} taken from --array" if args.array else
               "PE array 3x3 is only a DEFAULT: pass --array RxC (and --clk-mhz)")
        return fallback, "ASSUMED, not reported by the FPGA", [why, arr]


# ---------------------------------------------------------------- helpers
def u8(a):
    return bytes(int(v) & 0xFF for v in a)


def stats(a):
    a = np.asarray(a)
    return f"{a.size:,} values, min={int(a.min())}, max={int(a.max())}, crc32=0x{zlib.crc32(u8(a)) & 0xFFFFFFFF:08x}"


def ms(cycles, clk_hz):
    return cycles / clk_hz * 1e3


def layer_plan(L, hw):
    """Tiling the layer controller will use on THIS array (mirrors conv1d_layer_top)."""
    rows = (2 if hw.rows >= 2 else 1) if L["pool"] else hw.rows      # pool needs pairs of output positions
    row_tiles = math.ceil(L["out_len"] / rows)
    col_tiles = math.ceil(L["cout"] / hw.cols)
    return dict(rows=rows, row_tiles=row_tiles, col_tiles=col_tiles, tiles=row_tiles * col_tiles,
                macs=L["out_len"] * L["cout"] * L["k"])


def out_shape(L):
    """(T, C) of the output after pooling and padding for the next layer."""
    t = L["out_len"] // 2 if L["pool"] else L["out_len"]
    return t + 2 * L["pad_next"], L["cout"]


def fit_problems(L, hw, x_len, T_out, C_out):
    """Things this layer needs that the synthesized hardware cannot provide."""
    p = []
    if L["cout"] > hw.cout_max:
        p.append(f"Cout {L['cout']} > hardware COUT_MAX {hw.cout_max}")
    if L["k"] > hw.feeder_depth:
        p.append(f"K {L['k']} > hardware FEEDER_DEPTH {hw.feeder_depth}")
    if len(L["weight"]) > hw.wgt_depth:
        p.append(f"weights {len(L['weight']):,} B > weight SRAM {hw.wgt_depth:,} B")
    if x_len and x_len > hw.ifmap_depth:
        p.append(f"input {x_len:,} B > ifmap SRAM {hw.ifmap_depth:,} B")
    if not L["gap"] and T_out * C_out > hw.ifmap_depth:
        p.append(f"output {T_out * C_out:,} B > ifmap SRAM {hw.ifmap_depth:,} B")
    if L["gap"] and hw.cout_max * 2 > hw.max_payload:
        p.append(f"GAP reply {hw.cout_max * 2} B > payload limit {hw.max_payload} B")
    return p


def decode_link_error(e, hw):
    """Turn a raw error into an explanation plus things to check."""
    code = nack_code(e.msg)
    if code is not None:
        return [f"FPGA rejected {e.name} (NACK code {code}): {NACK_CODES.get(code, 'unknown')}"] + {
            1: ["Corrupted packet: retry, check the UART wiring and baud rate."],
            2: ["Check the arguments just sent (length, address, configuration) against layer_cmd_controller.v."],
            3: ["Wrong command order: every layer must go WRITE_W -> SWAP -> WRITE_BIAS -> WRITE_SHIFT -> CONFIG -> RUN."],
            4: ["Check that the .sof of ecg_accel_top (layer protocol) is loaded, not an older tile-protocol design."],
        }.get(code, [])
    if "timeout" in e.msg.lower():
        return [f"FPGA did not answer {e.name} in time.",
                f"Check: (1) the right .sof is loaded, (2) COM port and baud ({hw.baud}),",
                "       (3) press reset on the board and retry, (4) UART_RX/UART_TX pin assignments."]

    if "CRC" in e.msg or "Sequence" in e.msg:
        return [f"Reply to {e.name} has a frame/CRC/sequence error: {e.msg}",
                "Usually line noise, or the FPGA is still sending an old packet: press reset and rerun."]
    return [f"Protocol error while sending {e.name}: {e.msg}"]


# ---------------------------------------------------------------- layer runner
class LayerRunner:
    def __init__(self, dev, log, hw, dump_dir=None):
        self.dev = dev if isinstance(dev, TracedDevice) else TracedDevice(dev, log)
        self.log = log
        self.hw = hw
        self.dump_dir = dump_dir
        self.wchunk = hw.max_payload - 2                       # payload = addr:u16 + data
        self.bchunk = min(32, (hw.max_payload - 2) // 4)       # INT32 bias values per packet
        self.rchunk = min(250, hw.max_payload)                 # READ_FMAP bytes per packet

    # ---- layout helpers ----
    def kv(self, label, value, level=0):
        self.log.out(f"   {label:<12}{value}", level)

    def sub(self, text, level=0):
        self.log.out(f"   {'':<12}{text}", level)

    STEP_HEAD = f"   {'':1}  {'Step':<38}{'Pkts':>5}{'TX B':>9}{'RX B':>9}{'Time':>10}"

    @contextmanager
    def phase(self, tag, title, detail=()):
        """One table row per step; data statistics are shown with -v."""
        d = self.dev
        for line in detail:
            self.log.out(f"        {tag}: {line}", 1)
        p0, tx0, rx0, t0 = d.packets, d.tx_bytes, d.rx_bytes, time.monotonic()
        yield
        self.log.out(f"   {tag}  {title:<38}{d.packets - p0:>5}{d.tx_bytes - tx0:>9,}{d.rx_bytes - rx0:>9,}"
                     f"{time.monotonic() - t0:>8.2f} s")

    def skipped(self, tag, title):
        self.log.out(f"   {tag}  {title:<38}{'-':>5}{'-':>9}{'-':>9}{'-':>10}")

    def write_bytes(self, cmd, data):
        for off in range(0, len(data), self.wchunk):
            part = data[off:off + self.wchunk]
            self.dev.exchange(cmd, struct.pack("<H", off) + u8(part), RSP_ACK,
                              note=f"addr {off}..{off + len(part) - 1}")

    def write_bias(self, bias):
        for off in range(0, len(bias), self.bchunk):
            part = bias[off:off + self.bchunk]
            self.dev.exchange(CMD_WRITE_BIAS, struct.pack("<H", off) + struct.pack(f"<{len(part)}i", *[int(v) for v in part]),
                              RSP_ACK, note=f"channels {off}..{off + len(part) - 1}")

    # ---- comparison ----
    def compare_map(self, name, got, exp, C, pad, rpt):
        """Compare a layer output (T x C, channel-last) with golden; describe where/how it differs."""
        W = lambda s: self.log.out(f"   {'':<12}{s}")          # noqa: E731 - detail line under "Verify"
        if len(got) != len(exp):
            self.kv("Verify", f"[FAIL] {name}: received {len(got)} B, expected {len(exp)} B (wrong size)")
            return False
        G = np.frombuffer(bytes(got), np.uint8).view(np.int8).reshape(-1, C).astype(np.int64)
        E = np.frombuffer(bytes(exp), np.uint8).view(np.int8).reshape(-1, C).astype(np.int64)
        T = G.shape[0]
        bad = G != E
        nb = int(bad.sum())
        if nb == 0:
            self.kv("Verify", f"[PASS] {G.size:,}/{G.size:,} B match golden "
                              f"({T} x {C}, {int((E != 0).sum()):,} non-zero)")
            return True

        D = G - E
        self.kv("Verify", f"[FAIL] {name}: {nb:,}/{G.size:,} B differ ({100.0 * nb / G.size:.1f}%), "
                          f"{T} positions x {C} channels")

        per_c = bad.sum(axis=0)
        bad_c = [int(c) for c in np.where(per_c > 0)[0]]
        if len(bad_c) == C:
            W(f"by channel : EVERY channel has errors (min {int(per_c.min())}, max {int(per_c.max())} per channel)")
        else:
            lst = ", ".join(f"c{c}:{int(per_c[c])}" for c in bad_c[:16]) + (" ..." if len(bad_c) > 16 else "")
            W(f"by channel : {len(bad_c)}/{C} channels have errors -> {lst}")
            good = [c for c in range(C) if per_c[c] == 0]
            W(f"             fully correct: {good if len(good) <= 16 else f'{len(good)} channels'}")

        ts = np.where(bad.any(axis=1))[0]
        in_pad = int(bad[[t for t in range(T) if t < pad or t >= T - pad]].sum()) if pad else 0
        W(f"by position: {len(ts)}/{T} positions have errors, first t={int(ts[0])}, last t={int(ts[-1])}"
          f"{f', {in_pad} of them in the padding' if pad else ''}")

        first = np.argwhere(bad)
        W("first mismatches:")
        for t, c in first[:6]:
            W(f"  t={int(t):4d} c={int(c):3d}   got={int(G[t, c]):5d}  exp={int(E[t, c]):5d}  diff={int(D[t, c]):+d}")

        t0 = int(first[0][0])
        w = min(C, 16)
        W(f"row t={t0} (channels 0..{w - 1}):")
        W("  got : " + " ".join(f"{int(v):4d}" for v in G[t0, :w]))
        W("  exp : " + " ".join(f"{int(v):4d}" for v in E[t0, :w]))
        W("  diff: " + " ".join("   ^" if b else "    " for b in bad[t0, :w]))

        hints = []
        if not G.any():
            hints.append("Output is ALL ZERO: fmap_writer wrote nothing, wrong bank read (no swap), or RUN did not really run.")
        else:
            for s in (1, -1, 2, -2):
                lo, hi = (s, T) if s > 0 else (0, T + s)
                if (G[lo:hi] == np.roll(E, s, axis=0)[lo:hi]).mean() > 0.9:
                    hints.append(f"Output MATCHES golden shifted by {s:+d} position(s) in t: wrong write address/order (out_pos, padding, pool).")
            if C > 1:
                for s in (1, -1):
                    if (G == np.roll(E, s, axis=1)).mean() > 0.9:
                        hints.append(f"Output MATCHES golden shifted by {s:+d} channel(s): wrong column/channel index on write.")
        if 0 < len(bad_c) < C:
            mods = {c % self.hw.cols for c in bad_c}
            if len(mods) == 1:
                hints.append(f"All errors in channels with c % {self.hw.cols} == {mods.pop()}: suspect ONE PE COLUMN "
                             f"(its weights/bias/shift) of the {self.hw.rows}x{self.hw.cols} array.")
        inner = [int(t) - pad for t in ts if pad <= t < T - pad]
        if rpt > 1 and len(inner) > 1 and len({v % rpt for v in inner}) == 1:
            hints.append(f"All errors in tile row {inner[0] % rpt} of {rpt}: suspect ONE PE ROW (or the max branch of the pool pair).")
        if int(np.abs(D[bad]).max()) <= 1:
            hints.append("Only +-1 off: suspect rounding (rshift_round), shift off by one, or bias.")
        if (G[bad] == 0).mean() > 0.8:
            hints.append("Most wrong bytes are got=0 (expected non-zero): ReLU/clamp too aggressive, or the writer skipped those positions.")
        if pad and in_pad == nb:
            hints.append("Errors only in the padding: wrong number of pad rows (row_pad) or zero-fill.")
        if hints:
            W("hints:")
            for h in hints:
                for i, ln in enumerate(textwrap.wrap(h, 58)):
                    W(("  - " if i == 0 else "    ") + ln)

        if self.dump_dir:
            p = Path(self.dump_dir) / f"{name.replace(' ', '_')}_got_exp.npz"
            np.savez(p, got=G, exp=E)
            W(f"got/exp arrays saved: {p}")
        return False

    def compare_gap(self, got, gap):
        exp = np.array([int(v) & 0xFFFF for v in gap], dtype=np.int64)
        G = np.array(struct.unpack(f"<{len(got) // 2}H", bytes(got)), dtype=np.int64)
        W = lambda s: self.log.out(f"   {'':<12}{s}")          # noqa: E731
        if len(G) != len(exp):
            self.kv("Verify", f"[FAIL] GAP: received {len(G)} channels, expected {len(exp)}")
            return False
        bad = np.where(G != exp)[0]
        if len(bad) == 0:
            self.kv("Verify", f"[PASS] {len(exp)}/{len(exp)} GAP channels match golden "
                              f"(values {int(exp.min())}..{int(exp.max())})")
            return True
        D = G - exp
        self.kv("Verify", f"[FAIL] GAP: {len(bad)}/{len(exp)} channels differ")
        for c in bad[:8]:
            W(f"  c={int(c):3d}   got={int(G[c]):6d}  exp={int(exp[c]):6d}  diff={int(D[c]):+d}")
        hints = []
        if not G.any():
            hints.append("GAP is all zero: writer GAP mode is not accumulating, or layer 3 did not run.")
        if len(set(D[bad].tolist())) == 1:
            hints.append(f"Every wrong channel is off by the same constant ({int(D[bad[0]]):+d}): suspect bias, or one position missing/extra.")
        if (np.abs(D[bad]) <= np.maximum(1, exp[bad] // 100)).all():
            hints.append("Small error (<1%): a few input values from layer 2 -> layer 3 are wrong, not the accumulation itself.")
        if hints:
            W("hints:")
            for h in hints:
                for i, ln in enumerate(textwrap.wrap(h, 58)):
                    W(("  - " if i == 0 else "    ") + ln)
        return False

    # ---- one layer ----
    def run_layer(self, n, L, x, expect, gap, first):
        d, hw = self.dev, self.hw
        plan = layer_plan(L, hw)
        T_out, C_out = out_shape(L)
        src = ("golden output of layer %d" % (n - 1) if (first and n) else
               "x0 (z_q + zero pad)" if first else "output of layer %d (already in SRAM)" % (n - 1))
        dst = (f"GAP, {L['cout']} channels (u16)" if L["gap"] else
               f"{T_out} x {C_out} = {T_out * C_out:,} B (with padding)")
        self.log.out("")
        self.log.out("-" * WIDTH)
        self.log.out(f" LAYER {n}   conv{n}")
        self.log.out("-" * WIDTH)
        self.kv("Shape", f"Cin {L['cin']} -> Cout {L['cout']}, Kw {L['kw']} (K={L['k']}), out_len {L['out_len']}, "
                         f"{'pool 2' if L['pool'] else 'GAP' if L['gap'] else 'no pool'}, pad {L['pad_next']}")
        self.kv("Tiling", f"{plan['rows']} rows x {hw.cols} cols/tile, {plan['row_tiles']} x {plan['col_tiles']} "
                          f"= {plan['tiles']} tiles, rows used {plan['rows']}/{hw.rows}")
        self.kv("Work", f"{plan['macs']:,} MACs")
        self.kv("Memory", f"weights {len(L['weight']):,}/{hw.wgt_depth:,} B | K {L['k']}/{hw.feeder_depth} | "
                          f"Cout {L['cout']}/{hw.cout_max}")
        self.kv("Input", src)
        self.kv("Output", dst)

        problems = fit_problems(L, hw, len(x) if first else 0, T_out, C_out)
        if problems:
            self.log.out("")
            self.log.out(f"   [ABORT] layer {n} does not fit this hardware:")
            for p in problems:
                self.log.out(f"           - {p}")
            return False, 0, plan
        self.log.out("")
        self.log.out(self.STEP_HEAD)

        try:
            # [A] input
            if first:
                cin = L["cin"]
                with self.phase("A", "Load input X -> ifmap SRAM, swap",
                                [f"source: {'x0 = z_q (360) + 3 zeros each side (pad for K=7)' if n == 0 else f'golden output of layer {n - 1} (padding included)'}",
                                 f"shape : {len(x) // cin} positions x {cin} channels = {len(x):,} B | {stats(x)}"]):
                    self.write_bytes(CMD_WRITE_X, x)
                    d.exchange(CMD_SWAP, bytes((1,)), RSP_ACK, note="x_swap (bit0)")
            else:
                self.skipped("A", "Input already in ifmap SRAM")

            # [B] weights
            with self.phase("B", "Load weights -> weight SRAM, swap",
                            [f"shape : [Cout={L['cout']}][K={L['k']}] with K = j*Cin + i = {len(L['weight']):,} B",
                             stats(L["weight"]) + f" | filter 0, first 8: {[int(v) for v in L['weight'][:8]]}"]):
                self.write_bytes(CMD_WRITE_W, L["weight"])
                d.exchange(CMD_SWAP, bytes((2,)), RSP_ACK, note="w_swap (bit1)")

            # [C] bias + shift
            sh = [int(v) for v in L["shift"]]
            with self.phase("C", "Load bias (INT32) + shift per channel",
                            [f"bias : {stats(L['bias'])}",
                             f"shift: {len(sh)} channels, min={min(sh)}, max={max(sh)} -> {sh if len(sh) <= 16 else sh[:16] + ['...']}"]):
                self.write_bias(L["bias"])
                self.write_bytes(CMD_WRITE_SHIFT, L["shift"])

            # [D] config
            cfg = struct.pack("<HHHHBBB", L["out_len"], L["cout"], L["k"], L["step"], L["pool"], L["gap"], L["pad_next"])
            with self.phase("D", "CONFIG layer",
                            [f"out_len={L['out_len']} cout={L['cout']} k_len={L['k']} window_step={L['step']} (= Cin) "
                             f"pool={L['pool']} gap={L['gap']} pad_next={L['pad_next']}"]):
                d.exchange(CMD_CONFIG, cfg, RSP_ACK, note="11-byte config")

            # [E] run
            with self.phase("E", "RUN (FPGA runs all tiles, then swaps)"):
                res = d.exchange(CMD_RUN, b"", RSP_RESULT, expect=bytes(5), cmp_len=1, note="wait for layer done")
                status = res[0]
                cycles = struct.unpack("<I", res[1:5])[0] if len(res) == 5 else 0

            # [F] read
            ok = status == 0
            if n < 3:
                exp = u8(expect[n])
                got_all = bytearray()
                with self.phase("F", f"Read output ({T_out} x {C_out}) from ifmap SRAM",
                                [f"{len(exp):,} B in packets of up to {self.rchunk} B"]):
                    for off in range(0, len(exp), self.rchunk):
                        ln = min(self.rchunk, len(exp) - off)
                        got_all += d.exchange(CMD_READ_FMAP, struct.pack("<HB", off, ln), RSP_RESULT,
                                              expect=exp[off:off + ln], note=f"offset {off}..{off + ln - 1}")
            else:
                exp = b"".join(struct.pack("<H", int(v) & 0xFFFF) for v in gap)
                with self.phase("F", "Read GAP (64 x u16)"):
                    got = d.exchange(CMD_READ_GAP, b"", RSP_RESULT, expect=exp, note=f"{len(exp)} B")
        except LinkError as e:
            self.log.out("")
            self.log.out(f"   [LINK ERROR] at {e.name} {e.note}")
            for line in decode_link_error(e, hw):
                self.log.out(f"      {line}")
            return False, 0, plan

        self.log.out("")
        util = 100.0 * plan["macs"] / (cycles * hw.pes) if cycles else 0.0
        self.kv("Run", f"status {status} ({'OK' if status == 0 else 'writer OVERRUN'}) | {cycles:,} cycles = "
                       f"{ms(cycles, hw.clk_hz):.2f} ms @ {hw.clk_hz / 1e6:.2f} MHz")
        self.kv("Efficiency", f"{cycles / plan['tiles']:.1f} cycles/tile | PE util {util:.1f}% "
                              f"({plan['macs']:,} MACs / {cycles * hw.pes / 1e6:.1f}M slots)")
        if status != 0:
            self.sub("writer overrun: the fmap writer overflowed (next tile started before the previous one was written).")
        if n < 3:
            ok = self.compare_map(f"layer{n}_fmap", bytes(got_all), exp, C_out, L["pad_next"],
                                  plan["rows"] // (2 if L["pool"] else 1)) and ok
        else:
            ok = self.compare_gap(got, gap) and ok
        self.log.out(f" LAYER {n}: {'PASS' if ok else 'FAIL'}")
        return ok, cycles, plan

    # ---- all layers ----
    def run(self, model, x0, expect, gap, last_layer=3, first_layer=0, keep_going=False):
        hw = self.hw
        pe = f"PE {hw.rows}x{hw.cols}"
        self.log.out("")
        self.log.out(" Control flow (per layer)")
        self.log.out("   WRITE_W > SWAP > WRITE_BIAS > WRITE_SHIFT > CONFIG > RUN > READ")
        self.log.out(" Data flow")
        self.log.out("   host --UART--> weight/ifmap SRAM (write bank) --swap--> read bank")
        self.log.out(f"   --> feeder --> {pe} --> post (+bias,>>shift,ReLU,clamp) --> [maxpool 2]")
        self.log.out("   --> fmap_writer --> ifmap SRAM (write bank, + pad) --swap--> next layer")
        results = []
        t_all = time.monotonic()
        for n in range(first_layer, last_layer + 1):
            first = n == first_layer
            x = (x0 if n == 0 else expect[n - 1]) if first else None
            ok, cycles, plan = self.run_layer(n, model[n], x, expect, gap, first)
            results.append((n, ok, cycles, plan))
            if not ok and not keep_going and n < last_layer:
                self.log.out("")
                self.log.out(f" STOPPED at layer {n}: later layers would run on wrong data.")
                self.log.out(f" Use --first-layer {n} to test layer {n} alone with golden input,")
                self.log.out(" or --keep-going to run the remaining layers anyway.")
                break
        return self.summary(model, results, time.monotonic() - t_all, first_layer, last_layer)

    def summary(self, model, results, wall, first_layer, last_layer):
        expected_layers = last_layer - first_layer + 1
        hw = self.hw
        self.log.out("")
        self.log.out("=" * WIDTH)
        self.log.out(f" SUMMARY{f'PE array {hw.rows}x{hw.cols} @ {hw.clk_hz / 1e6:.2f} MHz':>{WIDTH - 8}}")
        self.log.out("=" * WIDTH)
        self.log.out(f" {'Layer':<6}{'Conv':<15}{'Result':<8}{'Cycles':>10}{'Time':>11}{'Tiles':>7}{'Cyc/tile':>10}{'Util':>7}")
        self.log.out(" " + "-" * (WIDTH - 2))
        tot_c = tot_m = tot_t = 0
        for n, ok, cycles, plan in results:
            L = model[n]
            tot_c += cycles
            tot_m += plan["macs"] if cycles else 0
            tot_t += plan["tiles"] if cycles else 0
            conv = f"{L['cin']}->{L['cout']}, K={L['k']}"
            if cycles:
                self.log.out(f" {n:<6}{conv:<15}{'PASS' if ok else 'FAIL':<8}{cycles:>10,}{ms(cycles, hw.clk_hz):>8.2f} ms"
                             f"{plan['tiles']:>7}{cycles / plan['tiles']:>10.1f}{100.0 * plan['macs'] / (cycles * hw.pes):>6.1f}%")
            else:
                self.log.out(f" {n:<6}{conv:<15}{'FAIL':<8}{'-':>10}{'-':>11}{'-':>7}{'-':>10}{'-':>7}")
        all_ok = all(ok for _, ok, _, _ in results) and len(results) == expected_layers
        self.log.out(" " + "-" * (WIDTH - 2))
        if tot_c:
            self.log.out(f" {'Total':<6}{'':<15}{'PASS' if all_ok else 'FAIL':<8}{tot_c:>10,}{ms(tot_c, hw.clk_hz):>8.2f} ms"
                         f"{tot_t:>7}{tot_c / tot_t:>10.1f}{100.0 * tot_m / (tot_c * hw.pes):>6.1f}%")
        self.log.out(f" Util = MACs / (cycles x {hw.pes} PEs); Cyc/tile = cycles per output tile.")
        self.log.out("")
        if tot_c and first_layer == 0 and last_layer == len(model) - 1 and all_ok:
            self.kv("Compute", f"{ms(tot_c, hw.clk_hz):.2f} ms per beat = {1e3 / ms(tot_c, hw.clk_hz):.1f} beats/s "
                               f"(UART transfer excluded)")
        elif tot_c:
            self.kv("Compute", f"{ms(tot_c, hw.clk_hz):.2f} ms for layers {first_layer}..{last_layer} (partial run)")
        line = hw.baud / 10 / 1e3
        eff = (self.dev.tx_bytes + self.dev.rx_bytes) / wall / 1e3 if wall > 0 else 0.0
        self.kv("Host link", f"{self.dev.packets} packets | {self.dev.tx_bytes:,} B sent | {self.dev.rx_bytes:,} B received | "
                             f"wall {wall:.1f} s")
        self.sub(f"payload {eff:.1f} kB/s vs {line:.1f} kB/s UART line rate")
        return all_ok


def banner(log, title):
    log.out("=" * WIDTH)
    log.out(f" {title}")
    log.out("=" * WIDTH)


def hw_report(log, hw, source, notes=()):
    log.out(f" Hardware  [{source}]")
    for note in notes:
        log.out(f"   ! {note}")
    log.out(f"   {'PE array':<13}{hw.rows} x {hw.cols}  ({hw.pes} PEs, {hw.pes} MAC/cycle peak)")
    log.out(f"   {'Clock':<13}{hw.clk_hz / 1e6:.2f} MHz")
    log.out(f"   {'UART':<13}{hw.baud} baud  ({hw.baud / 10 / 1e3:.1f} kB/s line rate, 8N1)")
    log.out(f"   {'Ifmap SRAM':<13}{hw.ifmap_depth:,} B per bank, 2 banks (ping-pong)")
    log.out(f"   {'Weight SRAM':<13}{hw.wgt_depth:,} B per bank, 2 banks")
    log.out(f"   {'Limits':<13}Cout <= {hw.cout_max}, K <= {hw.feeder_depth}, payload <= {hw.max_payload} B/packet")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("port", nargs="?", help="COM port, e.g. COM4")
    ap.add_argument("--baud", type=int, default=115200, help="host-side baud rate (must match the FPGA)")

    ap.add_argument("--timeout", type=float, default=5.0)
    ap.add_argument("--root", help="repo root (contains export/weights.h); default: parent of host/")
    ap.add_argument("--record", default="rec101_93600")
    ap.add_argument("--beat", type=int, default=0)
    ap.add_argument("--first-layer", type=int, default=0, choices=range(4),
                    help="load golden output of layer N-1 as input and start at layer N (test one layer alone)")
    ap.add_argument("--last-layer", type=int, default=3, choices=range(4))
    ap.add_argument("--keep-going", action="store_true", help="keep running later layers after a failed layer")
    ap.add_argument("--array", type=parse_array, metavar="RxC",
                    help="PE array size, ONLY used when the FPGA cannot report it (old bitstream) or with --dry-run")
    ap.add_argument("--clk-mhz", type=float, default=50.0,
                    help="clock in MHz, ONLY used when the FPGA cannot report it or with --dry-run")
    ap.add_argument("-v", "--verbose", action="count", default=0, help="-v: data statistics + one line per packet; -vv: payload bytes")
    ap.add_argument("--log-dir", help="log directory (default: results/hardware_logs)")
    ap.add_argument("--no-log", action="store_true", help="do not write a log file")
    ap.add_argument("--dry-run", metavar="TRACE", help="write the packet sequence to a hex file for a testbench (no board needed)")
    args = ap.parse_args()
    if args.root:
        global ROOT
        ROOT = Path(args.root)
    if args.first_layer > args.last_layer:
        ap.error("--first-layer must be <= --last-layer")

    model = load_model()
    x0, expect, gap = load_beat(args.record, args.beat)
    if args.dry_run:
        dev = DryRunDevice()
        log = Log(None, verbose=-1)
        hw, _, _ = query_hw(dev, log, args)
        LayerRunner(dev, log, hw).run(model, x0, expect, gap, args.last_layer, args.first_layer)
        size = dev.write_trace(args.dry_run)
        print(f"{len(dev.packets)} packets, {size} bytes -> {args.dry_run}")
        return 0
    if not args.port:
        ap.error("a COM port or --dry-run is required")

    log_path = dump_dir = None
    if not args.no_log:
        dump_dir = Path(args.log_dir) if args.log_dir else ROOT / "results" / "hardware_logs"
        dump_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_path = dump_dir / f"layer{args.first_layer}-{args.last_layer}_{args.record}_beat{args.beat:03d}_{stamp}.log"
    log = Log(log_path, verbose=args.verbose)

    banner(log, "ECG CNN1D ACCELERATOR - LAYER TEST")
    log.out(" Run")
    log.out(f"   {'Date':<13}{datetime.now():%Y-%m-%d %H:%M:%S}")
    log.out(f"   {'Port':<13}{args.port} @ {args.baud} baud")
    log.out(f"   {'Input':<13}record {args.record}, beat {args.beat}")
    log.out(f"   {'Layers':<13}{args.first_layer}..{args.last_layer}"
            f"{' (golden input for layer %d)' % args.first_layer if args.first_layer else ''}")
    if log_path:
        log.out(f"   {'Log file':<13}{log_path}")
    try:
        dev = SerialDevice(args.port, args.baud, args.timeout)
    except Exception as e:                           # noqa: BLE001
        log.out("")
        log.out(f" Could not open {args.port}: {e}")
        log.out(" Check the COM port name (Device Manager), that no other program holds it, and the USB cable.")
        log.close()
        return 2
    try:
        hw, source, notes = query_hw(dev, log, args)
        log.out("")
        hw_report(log, hw, source, notes)
        ok = LayerRunner(dev, log, hw, dump_dir).run(model, x0, expect, gap, args.last_layer,
                                                     args.first_layer, args.keep_going)
    finally:
        dev.close()
    log.out("")
    log.out(f" RESULT: {'PASS' if ok else 'FAIL'}")
    log.out("=" * WIDTH)
    log.close()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
