"""Run whole CNN layers on the FPGA (ecg_accel_top) and compare against golden data.

Layers: 0..3 = conv0..conv3 (conv3 ends in GAP), 4 = fc1, 5 = fc2 (INT32 logits + argmax on the FPGA).
FC runs on the same PE array as conv (FC = conv with Kw = 1), selected by CONFIG flags.

Two modes:
  python ecg_layer_host.py COM4 [--record rec101_93600 --beat 0 --last-layer 5]
        Connect over UART, query the hardware configuration (INFO), run layers
        0..last-layer and compare every layer with the golden output.
  python ecg_layer_host.py COM4 --beats 0:20 [--record rec118_237600] | --all [--check-layers]
        Multi-beat mode: conv0..fc2 for many beats, only the final 17 B (logits + argmax)
        read back unless --check-layers; summary = bit-exact count, confusion matrix vs the
        true labels, time per beat; one CSV row per beat next to the log. --sim: no board.
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
import csv
import json
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
from functools import lru_cache
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

# ---- CONFIG flags (12th CONFIG byte, see conv1d_layer_top.v) ----
F_OUT_INT32, F_ARGMAX, F_ROW_COMBINE, F_GAP_IN = 0x1, 0x2, 0x4, 0x8
FLAG_NAMES = {F_OUT_INT32: "out_int32", F_ARGMAX: "argmax", F_ROW_COMBINE: "row_combine", F_GAP_IN: "gap_in"}
F_BASES = 0x10            # INFO feature (not a CONFIG flag): CONFIG takes w_base / b_base -> weights resident on chip
BIAS_DEPTH = 256          # bias RAM entries of that RTL (conv1d_layer_top BIAS_DEPTH)
BANK_A, BANK_B = (0, 1, 2, 4, 5), (3,)   # resident layout: conv3 (12 KB) alone in bank B, the rest in bank A
CLASSES = "NSVF"

# ---- CNN Lite model: (Cout, Cin, Kw, pool, gap, pad_next, out_len, flags) ----
# FC = conv with Kw = 1. FC1 uses 3 PE rows of one tile (GAP low byte, GAP high byte, RR) that
# row_combine merges into one output position; FC2 keeps raw INT32 logits and stores the argmax.
LAYERS = [
    dict(name="conv0", cout=16, cin=1,  kw=7, pool=1, gap=0, pad_next=2, out_len=360, flags=0),
    dict(name="conv1", cout=32, cin=16, kw=5, pool=1, gap=0, pad_next=1, out_len=180, flags=0),
    dict(name="conv2", cout=64, cin=32, kw=3, pool=1, gap=0, pad_next=1, out_len=90,  flags=0),
    dict(name="conv3", cout=64, cin=64, kw=3, pool=0, gap=1, pad_next=0, out_len=45,  flags=0),
    dict(name="fc1",   cout=32, cin=68, kw=1, pool=0, gap=0, pad_next=0, out_len=3,
         flags=F_ROW_COMBINE | F_GAP_IN),
    dict(name="fc2",   cout=4,  cin=32, kw=1, pool=0, gap=0, pad_next=0, out_len=1,
         flags=F_OUT_INT32 | F_ARGMAX),
]
N_LAYERS = len(LAYERS)
FC1, FC2 = 4, 5
POOL_SHAPES = [(16, 180), (32, 90), (64, 45)]            # (C, L) after pool of layer 0..2
GAP_C = LAYERS[3]["cout"]                                 # GAP channels feeding FC1
FC1_AUX_ADDR = 2 * LAYERS[FC1]["cin"] + GAP_C             # RR bytes: row 2, after the GAP part (= 200)


def c_array(header: str, ctype: str, name: str) -> np.ndarray:
    m = re.search(rf"static\s+const\s+{ctype}\s+{name}\s*\[[^=]+\]\s*=\s*\{{(.*?)\}};", header, re.S)
    if m is None:
        raise RuntimeError(f"{name} not found in weights.h")
    return np.array([int(v) for v in re.findall(r"-?\d+", m.group(1))], dtype=np.int64)


def load_model():
    header = (ROOT / "export" / "weights.h").read_text(encoding="utf-8")
    model = []
    for n, L in enumerate(LAYERS[:4]):
        w = c_array(header, "int8_t", f"CONV{n}_W").reshape(L["cout"], L["cin"], L["kw"])
        w_acc = w.transpose(0, 2, 1).reshape(L["cout"], L["kw"] * L["cin"])    # [o][j][i] -> [Cout][K]
        model.append(dict(L, k=L["kw"] * L["cin"], step=L["cin"], weight=w_acc.reshape(-1),
                          bias=c_array(header, "int32_t", f"CONV{n}_B"),
                          shift=c_array(header, "int8_t", f"CONV{n}_SHIFT"), preshift=None))
    wg = c_array(header, "int8_t", "FC1_WG").reshape(32, GAP_C)
    wr = c_array(header, "int8_t", "FC1_WR").reshape(32, 4)
    if c_array(header, "int8_t", "FC1_SHIFT_R").any():
        raise RuntimeError("FC1_SHIFT_R must be all 0: row_combine adds the RR block without a shift")
    L = LAYERS[FC1]
    model.append(dict(L, k=L["cin"], step=L["cin"], weight=np.concatenate([wg, wr], axis=1).reshape(-1),
                      bias=c_array(header, "int32_t", "FC1_B"),
                      shift=c_array(header, "int8_t", "FC1_SHIFT_OUT"),          # SO: final requant
                      preshift=c_array(header, "int8_t", "FC1_SHIFT_G")))        # SG: GAP block, before + RR
    L = LAYERS[FC2]
    model.append(dict(L, k=L["cin"], step=L["cin"], weight=c_array(header, "int8_t", "FC2_W"),
                      bias=c_array(header, "int32_t", "FC2_B"),
                      shift=np.zeros(L["cout"], np.int64), preshift=None))          # INT32 out: no requant
    return model


def rshift_round(acc, s):
    s = np.asarray(s, dtype=np.int64)
    return np.where(s > 0, (acc + (np.int64(1) << np.maximum(s - 1, 0))) >> s, acc)


def fc_forward(model, gap, rr):
    """FC1 + FC2 + argmax exactly as python/quant/int8.py (the golden)."""
    f1, f2 = model[FC1], model[FC2]
    w1 = f1["weight"].reshape(f1["cout"], f1["k"])
    acc_g = w1[:, :GAP_C] @ np.asarray(gap, np.int64)
    acc_r = w1[:, GAP_C:] @ np.asarray(rr, np.int64)
    h = np.clip(rshift_round(rshift_round(acc_g, f1["preshift"]) + acc_r + f1["bias"], f1["shift"]), 0, 127)
    logits = f2["weight"].reshape(f2["cout"], f2["k"]) @ h + f2["bias"]
    return h, logits, int(np.argmax(logits))                 # np.argmax: ties -> smallest index (= RTL)


@lru_cache(maxsize=None)
def _golden_file(record: str, name: str, dtype):
    """Whole golden array of one segment, read once (multi-beat runs index it many times)."""
    return np.fromfile(ROOT / "export" / "golden" / f"{record}_beat_{name}.bin", dtype=dtype)


def golden_segments():
    """[(record, n_beats)] from export/golden/manifest.json."""
    m = json.loads((ROOT / "export" / "golden" / "manifest.json").read_text(encoding="utf-8"))
    return [(s["name"], int(s["n_beats"])) for s in m["segments"]]


def load_beat(record: str, beat: int):
    def arr(name, dtype, shape):
        return _golden_file(record, name, dtype).reshape(shape)[beat]

    z = arr("z_q", np.int8, (-1, 360)).astype(np.int64)
    x0 = np.concatenate([np.zeros(3, np.int64), z, np.zeros(3, np.int64)])    # padding for K=7
    expect = []
    for n, (shape, L) in enumerate(zip(POOL_SHAPES, LAYERS[:3])):
        p = arr(f"pool{n}", np.int8, (-1, *shape)).astype(np.int64).T          # (L, C)
        pad = L["pad_next"]
        expect.append(np.pad(p, ((pad, pad), (0, 0))).reshape(-1))
    gap = arr("gap", np.int32, (-1, 64)).astype(np.int64)
    fc = dict(rr=arr("rr_q", np.int8, (-1, 4)).astype(np.int64),
              h=arr("fc1", np.int8, (-1, 32)).astype(np.int64),
              logits=arr("logits", np.int32, (-1, 4)).astype(np.int64),
              pred=int(arr("pred", np.int8, (-1,))), label=int(arr("label", np.int8, (-1,))))
    return x0, expect, gap, fc


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
    features: int = 0                      # INFO byte 3 (version >= 2): CONFIG flags the bitstream supports

    @property
    def pes(self):
        return self.rows * self.cols

    @classmethod
    def from_payload(cls, p: bytes):
        if len(p) != INFO_SIZE:
            raise ValueError(f"INFO reply is {len(p)} B, expected {INFO_SIZE} B")
        ver, rows, cols, feat, clk, baud, ifm, wgt, feed, cmax, mp = struct.unpack(INFO_FMT, p)
        return cls(rows, cols, clk, baud, ifm, wgt, feed, cmax, mp, True, ver, feat if ver >= 2 else 0)

    @classmethod
    def assumed(cls, rows, cols, clk_hz, baud, features=0):
        # Remaining limits are the ecg_accel_top defaults.
        return cls(rows, cols, clk_hz, baud, 4096, 16384, 1024, 64, 256, False, 1, features)

    def to_payload(self):
        return struct.pack(INFO_FMT, self.version, self.rows, self.cols, self.features, self.clk_hz, self.baud,
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


class MemLog(Log):
    """Keeps the detailed per-layer report of the current beat in memory (multi-beat mode):
    nothing on screen, dumped into the main log only when that beat fails."""

    def __init__(self):
        super().__init__(None, verbose=-1)
        self.lines = []

    def out(self, msg="", level=0):
        self.lines.append(msg)


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

    def resync(self):
        """After a failed beat: let the FPGA finish / time out (RX timeout 100 ms), drop stale bytes."""
        time.sleep(0.3)
        self.acc.uart.reset_input_buffer()

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

    def resync(self):
        pass

    def close(self):
        pass


class SimDevice(DryRunDevice):
    """--sim: answers every request with the golden reply and keeps no packet list (multi-beat dry run)."""

    def exchange(self, cmd, payload, rsp, expect=None, cmp_len=None):
        if rsp == RSP_ACK:
            return bytes((cmd, 0))
        if cmd == CMD_RUN:
            return bytes((0,)) + struct.pack("<I", 1)
        return expect if expect is not None else b""


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
        fallback.features = F_OUT_INT32 | F_ARGMAX | F_ROW_COMBINE | F_GAP_IN | F_BASES   # trace for the current RTL
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
    rows = min(rows, L["out_len"])                                    # FC: 1 (fc2) or 3 (fc1) rows used
    row_tiles = math.ceil(L["out_len"] / rows)
    col_tiles = math.ceil(L["cout"] / hw.cols)
    return dict(rows=rows, row_tiles=row_tiles, col_tiles=col_tiles, tiles=row_tiles * col_tiles,
                macs=L["out_len"] * L["cout"] * L["k"])


def out_shape(L):
    """(T, C) of the output after pooling / row_combine and padding for the next layer."""
    if L["flags"] & F_OUT_INT32:
        return 1, 4 * L["cout"] + (1 if L["flags"] & F_ARGMAX else 0)   # LE int32 per channel [+ argmax byte]
    t = 1 if L["flags"] & F_ROW_COMBINE else L["out_len"] // 2 if L["pool"] else L["out_len"]
    return t + 2 * L["pad_next"], L["cout"]


def flag_text(flags):
    return "+".join(v for k, v in FLAG_NAMES.items() if flags & k) or "none"


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
    missing = L["flags"] & ~hw.features
    if missing:
        p.append(f"needs CONFIG flags {flag_text(missing)}, the bitstream reports {flag_text(hw.features)} "
                 f"(INFO version {hw.version}{'' if hw.reported else ', assumed'}): rebuild with the FC RTL")
    if L["flags"] & F_ROW_COMBINE and hw.rows < 3:
        p.append(f"row_combine needs >= 3 PE rows, the array has {hw.rows}")
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
        self.fast = False            # multi-beat: skip reading/comparing intermediate layers, read only fc2
        self.last_out = None         # (logits, pred) read back from the FPGA by the last fc2
        self.last_results = []       # [(n, ok, cycles, plan)] of the last run()
        self.resident = False        # weights + bias loaded once (CONFIG w_base / b_base), see enable_resident
        self.layout = None           # {layer: {bank, w_base, b_base}}
        self.w_read = None           # which logical bank (0 = A, 1 = B) the weight READ bank holds now
        self.loaded = False          # resident model is on the chip (cleared after a failure -> reload)

    # ---- resident weights ----
    def enable_resident(self, model):
        """Place every layer: bank A = conv0,1,2,fc1,fc2, bank B = conv3; bias packed in one RAM."""
        lay, b = {}, 0
        for bank, idxs in ((0, BANK_A), (1, BANK_B)):
            w = 0
            for n in idxs:
                lay[n] = dict(bank=bank, w_base=w)
                w += len(model[n]["weight"])
            if w > self.hw.wgt_depth:
                return f"bank {'AB'[bank]} needs {w:,} B > weight bank {self.hw.wgt_depth:,} B"
        for n in range(N_LAYERS):
            lay[n]["b_base"] = b
            b += model[n]["cout"]
        if b > BIAS_DEPTH:
            return f"{b} biases > bias RAM {BIAS_DEPTH}"
        self.layout, self.resident, self.loaded = lay, True, False
        return ""

    def load_resident(self, model):
        """Write the whole model once. Works from any bank state: ends with READ = bank A, WRITE = bank B."""
        d = self.dev
        self.log.out("")
        self.log.out(" Load model once (resident weights, CONFIG w_base / b_base)")
        self.log.out(self.STEP_HEAD)
        with self.phase("W", "Bank A: conv0,1,2 + fc1,2 -> swap",
                        [f"{model[n]['name']} @ {self.layout[n]['w_base']}" for n in BANK_A]):
            for n in BANK_A:
                self.write_bytes(CMD_WRITE_W, model[n]["weight"], base=self.layout[n]["w_base"])
            d.exchange(CMD_SWAP, bytes((2,)), RSP_ACK, note="w_swap: bank A readable")
        with self.phase("W", "Bank B: conv3 (stays in the write bank)"):
            for n in BANK_B:
                self.write_bytes(CMD_WRITE_W, model[n]["weight"], base=self.layout[n]["w_base"])
        with self.phase("W", "Bias of all layers -> bias RAM",
                        [f"{model[n]['name']} @ {self.layout[n]['b_base']}" for n in range(N_LAYERS)]):
            for n in range(N_LAYERS):
                self.write_bias(model[n]["bias"], base=self.layout[n]["b_base"])
        self.w_read, self.loaded = 0, True

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

    def write_bytes(self, cmd, data, base=0):
        for off in range(0, len(data), self.wchunk):
            part = data[off:off + self.wchunk]
            self.dev.exchange(cmd, struct.pack("<H", base + off) + u8(part), RSP_ACK,
                              note=f"addr {base + off}..{base + off + len(part) - 1}")

    def write_bias(self, bias, base=0):
        for off in range(0, len(bias), self.bchunk):
            part = bias[off:off + self.bchunk]
            self.dev.exchange(CMD_WRITE_BIAS, struct.pack("<H", base + off) + struct.pack(f"<{len(part)}i", *[int(v) for v in part]),
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

    def compare_logits(self, got):
        """FC2 output: 4 x INT32 LE logits + argmax byte, against golden logits / pred (and the true label)."""
        fc = self.fc
        n_out = len(fc["logits"])
        if len(got) != 4 * n_out + 1:
            self.kv("Verify", f"[FAIL] logits: received {len(got)} B, expected {4 * n_out + 1} B")
            return False
        G = np.array(struct.unpack(f"<{n_out}i", got[:4 * n_out]), dtype=np.int64)
        pred = got[4 * n_out]
        self.last_out = ([int(v) for v in G], int(pred))
        ok_l = bool((G == fc["logits"]).all())
        ok_p = pred == fc["pred"]
        exp_txt = "" if ok_l else f"  exp {[int(v) for v in fc['logits']]}"
        self.kv("Verify", f"[{'PASS' if ok_l else 'FAIL'}] logits (x2^-11): got {[int(v) for v in G]}{exp_txt}")
        self.sub(f"= {', '.join(f'{c} {v / 2048:+.3f}' for c, v in zip(CLASSES, G))}")
        self.kv("", f"[{'PASS' if ok_p else 'FAIL'}] argmax on FPGA = {pred} ({CLASSES[pred] if pred < 4 else '?'}), "
                    f"golden pred {fc['pred']} ({CLASSES[fc['pred']]})")
        self.sub(f"true label {CLASSES[fc['label']]}: the model {'is right' if fc['pred'] == fc['label'] else 'is WRONG'} "
                 f"on this beat (a model property, not a hardware error)")
        if ok_l and not ok_p:
            self.sub("logits match but the argmax byte does not: check the argmax compare/tie rule or the address 4*cout.")
        return ok_l and ok_p

    # ---- one layer ----
    def run_layer(self, n, L, x, expect, gap, first):
        d, hw = self.dev, self.hw
        plan = layer_plan(L, hw)
        T_out, C_out = out_shape(L)
        src = ("GAP of conv3 (on chip) + 4 RR bytes from the host" if n == FC1 else
               "golden output of layer %d" % (n - 1) if (first and n) else
               "x0 (z_q + zero pad)" if first else "output of layer %d (already in SRAM)" % (n - 1))
        dst = (f"GAP, {L['cout']} channels (u16)" if L["gap"] else
               f"{L['cout']} x INT32 logits + argmax byte = {C_out} B" if L["flags"] & F_OUT_INT32 else
               f"{T_out} x {C_out} = {T_out * C_out:,} B (with padding)")
        self.log.out("")
        self.log.out("-" * WIDTH)
        self.log.out(f" LAYER {n}   {L['name']}")
        self.log.out("-" * WIDTH)
        self.kv("Shape", f"Cin {L['cin']} -> Cout {L['cout']}, Kw {L['kw']} (K={L['k']}), out_len {L['out_len']}, "
                         f"{'pool 2' if L['pool'] else 'GAP' if L['gap'] else 'no pool'}, pad {L['pad_next']}"
                         f"{', flags ' + flag_text(L['flags']) if L['flags'] else ''}")
        if n == FC1:
            self.sub("3 PE rows per channel: GAP & 127 | GAP >> 7 | RR  -> row_combine -> 1 output")
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
            if n == FC1:
                rr = self.fc["rr"]
                with self.phase("A", "Write RR (4 B) -> ifmap write bank",
                                [f"rr_q = {[int(v) for v in rr]} at address {FC1_AUX_ADDR} (row 2, after the "
                                 f"{GAP_C} GAP slots); gap_in packs GAP into rows 0-1 and swaps"]):
                    self.write_bytes(CMD_WRITE_X, rr, base=FC1_AUX_ADDR)
            elif first:
                cin = L["cin"]
                with self.phase("A", "Load input X -> ifmap SRAM, swap",
                                [f"source: {'x0 = z_q (360) + 3 zeros each side (pad for K=7)' if n == 0 else f'golden output of layer {n - 1} (padding included)'}",
                                 f"shape : {len(x) // cin} positions x {cin} channels = {len(x):,} B | {stats(x)}"]):
                    self.write_bytes(CMD_WRITE_X, x)
                    d.exchange(CMD_SWAP, bytes((1,)), RSP_ACK, note="x_swap (bit0)")
            else:
                self.skipped("A", "Input already in ifmap SRAM")

            # [B] weights
            if self.resident:
                lay = self.layout[n]
                need = lay["bank"] != self.w_read
                with self.phase("B", f"Weights resident: bank {'AB'[lay['bank']]} @ {lay['w_base']}"
                                     f"{', swap' if need else ''}",
                                [f"loaded once at start; this layer reads w_base={lay['w_base']}, b_base={lay['b_base']}"]):
                    if need:
                        d.exchange(CMD_SWAP, bytes((2,)), RSP_ACK, note=f"w_swap -> bank {'AB'[lay['bank']]}")
                        self.w_read = lay["bank"]
            else:
                with self.phase("B", "Load weights -> weight SRAM, swap",
                                [f"shape : [Cout={L['cout']}][K={L['k']}] with K = j*Cin + i = {len(L['weight']):,} B",
                                 stats(L["weight"]) + f" | filter 0, first 8: {[int(v) for v in L['weight'][:8]]}"]):
                    self.write_bytes(CMD_WRITE_W, L["weight"])
                    d.exchange(CMD_SWAP, bytes((2,)), RSP_ACK, note="w_swap (bit1)")

            # [C] bias + shift (resident: bias already on chip; shifts are re-written, the shift RAM is per channel)
            sh = [int(v) for v in L["shift"]]
            int32_out = bool(L["flags"] & F_OUT_INT32)
            with self.phase("C", "Load shift per channel" if self.resident else "Load bias (INT32) + shift per channel",
                            [f"bias : {stats(L['bias'])}",
                             f"shift: {len(sh)} channels, min={min(sh)}, max={max(sh)} -> {sh if len(sh) <= 16 else sh[:16] + ['...']}"]):
                if not self.resident:
                    self.write_bias(L["bias"])
                if not (self.resident and int32_out):            # INT32 output does not use the shift
                    self.write_bytes(CMD_WRITE_SHIFT, L["shift"])
                if L["preshift"] is not None:                    # SG of row_combine, at shift address COUT_MAX + c
                    self.write_bytes(CMD_WRITE_SHIFT, L["preshift"], base=hw.cout_max)

            # [D] config (12th byte = flags, 13..16 = w_base, b_base; shorter forms for old bitstreams)
            cfg = struct.pack("<HHHHBBB", L["out_len"], L["cout"], L["k"], L["step"], L["pool"], L["gap"], L["pad_next"])
            if self.resident:
                cfg += struct.pack("<BHH", L["flags"], self.layout[n]["w_base"], self.layout[n]["b_base"])
            elif L["flags"]:
                cfg += bytes((L["flags"],))
            with self.phase("D", "CONFIG layer",
                            [f"out_len={L['out_len']} cout={L['cout']} k_len={L['k']} window_step={L['step']} (= Cin) "
                             f"pool={L['pool']} gap={L['gap']} pad_next={L['pad_next']} flags={flag_text(L['flags'])}"]):
                d.exchange(CMD_CONFIG, cfg, RSP_ACK, note=f"{len(cfg)}-byte config")

            # [E] run
            with self.phase("E", "RUN (FPGA runs all tiles, then swaps)"):
                res = d.exchange(CMD_RUN, b"", RSP_RESULT, expect=bytes(5), cmp_len=1, note="wait for layer done")
                status = res[0]
                cycles = struct.unpack("<I", res[1:5])[0] if len(res) == 5 else 0

            # [F] read
            ok = status == 0
            skip_read = self.fast and n != FC2
            if skip_read:
                self.skipped("F", "Read output (skipped: --fast)")
            elif n != 3:
                exp = (u8(expect[n]) if n < 3 else u8(self.fc["h"]) if n == FC1 else
                       struct.pack(f"<{L['cout']}i", *[int(v) for v in self.fc["logits"]]) + bytes((self.fc["pred"],)))
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
        if skip_read:
            self.kv("Verify", "not read (multi-beat fast mode: checked through the final logits)")
        elif n < 3 or n == FC1:
            ok = self.compare_map(f"layer{n}_fmap", bytes(got_all), exp, C_out, L["pad_next"],
                                  plan["rows"] // (2 if L["pool"] else 1) if n < 3 else 1) and ok
        elif n == FC2:
            ok = self.compare_logits(bytes(got_all)) and ok
        else:
            ok = self.compare_gap(got, gap) and ok
        self.log.out(f" LAYER {n}: {'PASS' if ok else 'FAIL'}")
        return ok, cycles, plan

    # ---- all layers ----
    def run(self, model, x0, expect, gap, fc, last_layer=N_LAYERS - 1, first_layer=0, keep_going=False):
        hw = self.hw
        self.fc = fc
        pe = f"PE {hw.rows}x{hw.cols}"
        self.log.out("")
        self.log.out(" Control flow (per layer)")
        self.log.out("   WRITE_W > SWAP > WRITE_BIAS > WRITE_SHIFT > CONFIG > RUN > READ")
        self.log.out(" Data flow")
        self.log.out("   host --UART--> weight/ifmap SRAM (write bank) --swap--> read bank")
        self.log.out(f"   --> feeder --> {pe} --> post (+bias,>>shift,ReLU,clamp) --> [maxpool 2]")
        self.log.out("   --> fmap_writer --> ifmap SRAM (write bank, + pad) --swap--> next layer")
        results = []
        self.last_results = results
        self.last_out = None
        t_all = time.monotonic()
        if self.resident and not self.loaded:
            try:
                self.load_resident(model)
            except LinkError as e:
                self.log.out(f"   [LINK ERROR] while loading the model: {e.name} {e.note}")
                for line in decode_link_error(e, hw):
                    self.log.out(f"      {line}")
                return self.summary(model, results, time.monotonic() - t_all, first_layer, last_layer)
        for n in range(first_layer, last_layer + 1):
            first = n == first_layer
            x = (x0 if n == 0 else fc["h"] if n == FC2 else expect[n - 1]) if first else None
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
        self.log.out(f" {'Layer':<6}{'Shape':<15}{'Result':<8}{'Cycles':>10}{'Time':>11}{'Tiles':>7}{'Cyc/tile':>10}{'Util':>7}")
        self.log.out(" " + "-" * (WIDTH - 2))
        tot_c = tot_m = tot_t = 0
        for n, ok, cycles, plan in results:
            L = model[n]
            tot_c += cycles
            tot_m += plan["macs"] if cycles else 0
            tot_t += plan["tiles"] if cycles else 0
            conv = f"{L['name']} {L['cin']}->{L['cout']}"
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


# ---------------------------------------------------------------- multi-beat mode
def parse_beats(text: str, n: int):
    """'all' | 'A:B' (end exclusive, either side optional) | 'A' -> list of beat indices < n."""
    if text == "all":
        return list(range(n))
    m = re.fullmatch(r"\s*(\d*)\s*:\s*(\d*)\s*", text)
    if m:
        a = int(m.group(1)) if m.group(1) else 0
        b = int(m.group(2)) if m.group(2) else n
        return list(range(a, min(b, n)))
    return [int(text)] if int(text) < n else []


def macro_f1(conf):
    """Macro-F1 over the classes that occur (as true label or as prediction), like sklearn's default."""
    f1, used = [], []
    for c in range(len(conf)):
        tp = conf[c][c]
        p = sum(conf[r][c] for r in range(len(conf)))
        t = sum(conf[c])
        prec = tp / p if p else 0.0
        rec = tp / t if t else 0.0
        f1.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
        if p or t:
            used.append(f1[-1])
    return (sum(used) / len(used) if used else 0.0), f1


def run_multi(dev, log, hw, model, targets, check_layers, csv_path, resident=True):
    """Run conv0..fc2 for every (record, beat) and summarize. Returns True when every beat is bit-exact."""
    mem = MemLog()
    runner = LayerRunner(dev, mem, hw)
    runner.fast = not check_layers
    why = runner.enable_resident(model) if resident and hw.features & F_BASES else "not supported / disabled"
    log.out(f"   {'Weights':<14}{'resident: loaded once, then only input + commands per beat' if runner.resident else f're-sent every beat ({why})'}")
    conf = [[0] * 4 for _ in range(4)]                       # rows: true label, cols: FPGA argmax
    per_rec = {}
    rows, failures = [], []
    n_exact = n_label_vs_gold = 0
    tot_cycles = 0
    t_start = time.monotonic()
    log.out("")
    log.out(f" {'#':>9}  {'record':<14}{'beat':>5}  true  gold  FPGA  {'result':<8}{'cycles':>11}{'wall':>8}")
    log.out(" " + "-" * (WIDTH - 2))
    done = 0
    try:
        for i, (rec, beat) in enumerate(targets):
            x0, expect, gap, fc = load_beat(rec, beat)
            mem.lines.clear()
            t0 = time.monotonic()
            err = ""
            try:
                ok = runner.run(model, x0, expect, gap, fc, FC2, 0)
            except Exception as e:                       # noqa: BLE001 - serial/protocol error: report, continue
                ok, err = False, f"{type(e).__name__}: {e}"
            wall = time.monotonic() - t0
            cycles = sum(c for _, _, c, _ in runner.last_results)
            got = runner.last_out
            pred = got[1] if got else -1
            exact = bool(ok and got and got[0] == [int(v) for v in fc["logits"]] and pred == fc["pred"])
            done += 1
            n_exact += exact
            tot_cycles += cycles
            if 0 <= pred < 4:
                conf[fc["label"]][pred] += 1
                n_label_vs_gold += pred == fc["pred"]
            r = per_rec.setdefault(rec, [0, 0])
            r[0] += 1
            r[1] += exact
            res = "PASS" if exact else ("LINK ERR" if not got else "FAIL")
            log.out(f" {i + 1:>4}/{len(targets):<4}  {rec:<14}{beat:>5}   {CLASSES[fc['label']]}     {CLASSES[fc['pred']]}     "
                    f"{CLASSES[pred] if 0 <= pred < 4 else '-'}   {res:<8}{cycles:>11,}{wall:>7.2f}s")
            rows.append(dict(record=rec, beat=beat, label=CLASSES[fc["label"]], golden_pred=CLASSES[fc["pred"]],
                             fpga_pred=CLASSES[pred] if 0 <= pred < 4 else "", bit_exact=int(exact),
                             logits=" ".join(map(str, got[0])) if got else "", golden_logits=" ".join(map(str, fc["logits"])),
                             cycles=cycles, wall_s=round(wall, 3), error=err))
            if not exact:
                failures.append((rec, beat))
                log.f and log.f.write("\n".join(["", f"   ---- detail of {rec} beat {beat} ----"] + mem.lines[-80:]
                                                + ([f"   exception: {err}"] if err else [])) + "\n")
                dev.resync()
                runner.loaded = False                    # bank state unknown after an error: reload the model
    except KeyboardInterrupt:
        log.out("")
        log.out(f" Interrupted by the user after {done} beat(s): the summary covers those only.")
    wall_all = time.monotonic() - t_start

    if csv_path and rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    log.out("")
    log.out("=" * WIDTH)
    log.out(f" MULTI-BEAT SUMMARY{f'PE array {hw.rows}x{hw.cols} @ {hw.clk_hz / 1e6:.2f} MHz':>{WIDTH - 19}}")
    log.out("=" * WIDTH)
    if not done:
        log.out(" No beat was run.")
        return False
    log.out(f"   {'Hardware':<14}{n_exact}/{done} beats bit-exact with the golden (logits + argmax)"
            f"{'' if n_exact == done else f'  -> {done - n_exact} FAILED'}")
    n_lab = sum(map(sum, conf))                               # beats that returned a label (no link error)
    log.out(f"   {'Argmax':<14}{n_label_vs_gold}/{n_lab} FPGA labels = golden model labels"
            f"{f' ({done - n_lab} beat(s) without a label: link error)' if n_lab < done else ''}")
    acc = sum(conf[c][c] for c in range(4))
    mf1, f1s = macro_f1(conf)
    if n_lab:
        log.out(f"   {'Model':<14}{acc}/{n_lab} = {100.0 * acc / n_lab:.1f}% labels correct vs the TRUE label, "
                f"macro-F1 {mf1:.3f}")
    log.out(f"   {'':<14}(model property, not a hardware result: the golden segments are DS1 records the final")
    log.out(f"   {'':<14} model was trained on, so this is NOT a generalization score; see DS2: macro-F1 0.603)")
    log.out("")
    log.out(f"   Confusion (rows true, columns FPGA):   {'   '.join(CLASSES)}    recall   F1")
    for c in range(4):
        t = sum(conf[c])
        present = t or any(conf[r][c] for r in range(4))
        log.out(f"   {'':<24}{CLASSES[c]}    " + " ".join(f"{v:>4}" for v in conf[c])
                + f"   {f'{100.0 * conf[c][c] / t:5.1f}%' if t else '    - '}  {f'{f1s[c]:.3f}' if present else '  -'}")
    log.out("")
    log.out("   Per record:  " + "  ".join(f"{r} {p}/{n}" for r, (n, p) in per_rec.items()))
    log.out(f"   {'Compute':<14}{ms(tot_cycles / done, hw.clk_hz):.2f} ms per beat on the FPGA (average)")
    log.out(f"   {'Wall':<14}{wall_all / done:.2f} s per beat incl. UART ({wall_all / 60:.1f} min total)"
            f"{', intermediate layers not read back' if runner.fast else ''}")
    log.out(f"   {'Host link':<14}{runner.dev.packets:,} packets | {runner.dev.tx_bytes:,} B sent | "
            f"{runner.dev.rx_bytes:,} B received")
    if failures:
        shown = ", ".join(f"{r}#{b}" for r, b in failures[:12]) + (" ..." if len(failures) > 12 else "")
        log.out(f"   {'Failed beats':<14}{shown}")
        log.out(f"   {'':<14}details in the log file; rerun one with --record R --beat N (all layers checked)")
    if csv_path and rows:
        log.out(f"   {'CSV':<14}{csv_path}")
    return n_exact == done == len(targets)


def main_multi(args, model):
    segs = dict(golden_segments())
    if args.all:
        targets = [(r, b) for r, n in segs.items() for b in range(n)]
        tag = "all"
    else:
        if args.record not in segs:
            print(f"unknown record {args.record}; golden segments: {', '.join(segs)}", file=sys.stderr)
            return 2
        targets = [(args.record, b) for b in parse_beats(args.beats, segs[args.record])]
        tag = f"{args.record}_beats{args.beats.replace(':', '-')}"
    if not targets:
        print("no beat selected", file=sys.stderr)
        return 2
    # The host-side FC reference must reproduce the golden for every selected beat before judging the FPGA.
    for rec, b in targets:
        _, _, gap, fc = load_beat(rec, b)
        h_ref, lg_ref, pred_ref = fc_forward(model, gap, fc["rr"])
        if not ((h_ref == fc["h"]).all() and (lg_ref == fc["logits"]).all() and pred_ref == fc["pred"]):
            print(f"fc_forward() does not reproduce the golden at {rec} beat {b}", file=sys.stderr)
            return 2
    if not args.sim and not args.port:
        print("a COM port or --sim is required", file=sys.stderr)
        return 2

    log_path = csv_path = None
    if not args.no_log:
        out_dir = Path(args.log_dir) if args.log_dir else ROOT / "results" / "hardware_logs"
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        base = out_dir / f"multi_{tag}{'_sim' if args.sim else ''}_{stamp}"
        log_path, csv_path = base.with_suffix(".log"), base.with_suffix(".csv")
    log = Log(log_path, verbose=args.verbose)
    banner(log, "ECG CNN1D ACCELERATOR - MULTI-BEAT TEST (conv0 .. fc2 + argmax)")
    log.out(" Run")
    log.out(f"   {'Date':<13}{datetime.now():%Y-%m-%d %H:%M:%S}")
    log.out(f"   {'Port':<13}{'simulated FPGA (--sim)' if args.sim else f'{args.port} @ {args.baud} baud'}")
    log.out(f"   {'Beats':<13}{len(targets)} ({'all golden segments' if args.all else f'{args.record} [{args.beats}]'})")
    log.out(f"   {'Check':<13}{'every layer read back' if args.check_layers else 'final logits + argmax only (intermediate layers not read)'}")
    if log_path:
        log.out(f"   {'Log file':<13}{log_path}")
    try:
        dev = SimDevice() if args.sim else SerialDevice(args.port, args.baud, args.timeout)
    except Exception as e:                           # noqa: BLE001
        log.out(f" Could not open {args.port}: {e}")
        log.close()
        return 2
    try:
        hw, source, notes = query_hw(dev, log, args)
        log.out("")
        hw_report(log, hw, source, notes)
        missing = [L["name"] for L in model if L["flags"] & ~hw.features]
        if missing:
            log.out("")
            log.out(f" [ABORT] the bitstream does not support the CONFIG flags of {', '.join(missing)} "
                    f"(INFO features {flag_text(hw.features)}): rebuild with the FC RTL.")
            ok = False
        else:
            ok = run_multi(dev, log, hw, model, targets, args.check_layers, csv_path)
    finally:
        dev.close()
    log.out("")
    log.out(f" RESULT: {'PASS' if ok else 'FAIL'}")
    log.out("=" * WIDTH)
    log.close()
    return 0 if ok else 1


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("port", nargs="?", help="COM port, e.g. COM4")
    ap.add_argument("--baud", type=int, default=921600, help="host-side baud rate (must match the FPGA)")

    ap.add_argument("--timeout", type=float, default=5.0)
    ap.add_argument("--root", help="repo root (contains export/weights.h); default: parent of host/")
    ap.add_argument("--record", default="rec101_93600")
    ap.add_argument("--beat", type=int, default=0)
    ap.add_argument("--first-layer", type=int, default=0, choices=[0, 1, 2, 3, FC2],
                    help="load golden output of layer N-1 as input and start at layer N (test one layer alone); "
                         f"FC1 ({FC1}) reads the on-chip GAP, so it cannot start alone: use --first-layer 3")
    ap.add_argument("--last-layer", type=int, default=N_LAYERS - 1, choices=range(N_LAYERS),
                    help="0..3 = conv0..conv3, 4 = fc1, 5 = fc2 (logits + argmax)")
    ap.add_argument("--keep-going", action="store_true", help="keep running later layers after a failed layer")
    ap.add_argument("--array", type=parse_array, metavar="RxC",
                    help="PE array size, ONLY used when the FPGA cannot report it (old bitstream) or with --dry-run")
    ap.add_argument("--clk-mhz", type=float, default=50.0,
                    help="clock in MHz, ONLY used when the FPGA cannot report it or with --dry-run")
    ap.add_argument("-v", "--verbose", action="count", default=0, help="-v: data statistics + one line per packet; -vv: payload bytes")
    ap.add_argument("--log-dir", help="log directory (default: results/hardware_logs)")
    ap.add_argument("--no-log", action="store_true", help="do not write a log file")
    ap.add_argument("--dry-run", metavar="TRACE", help="write the packet sequence to a hex file for a testbench (no board needed)")
    ap.add_argument("--beats", metavar="A:B",
                    help="multi-beat mode on --record: 'all', 'A:B' (end exclusive) or one index; runs conv0..fc2 per beat")
    ap.add_argument("--all", action="store_true", help="multi-beat mode on every beat of every golden segment (1306 beats)")
    ap.add_argument("--check-layers", action="store_true",
                    help="multi-beat mode: also read back and compare every intermediate layer (slower)")
    ap.add_argument("--sim", action="store_true", help="multi-beat mode without a board: a simulated FPGA echoes the golden")
    args = ap.parse_args()
    if args.root:
        global ROOT
        ROOT = Path(args.root)
    if args.first_layer > args.last_layer:
        ap.error("--first-layer must be <= --last-layer")

    model = load_model()
    if args.beats or args.all:
        return main_multi(args, model)
    x0, expect, gap, fc = load_beat(args.record, args.beat)
    # The host-side FC reference must reproduce the golden before it is used to judge the FPGA.
    h_ref, lg_ref, pred_ref = fc_forward(model, gap, fc["rr"])
    if not ((h_ref == fc["h"]).all() and (lg_ref == fc["logits"]).all() and pred_ref == fc["pred"]):
        print("fc_forward() does not reproduce the golden FC1/FC2: weights.h and export/golden disagree", file=sys.stderr)
        return 2
    if args.dry_run:
        dev = DryRunDevice()
        log = Log(None, verbose=-1)
        hw, _, _ = query_hw(dev, log, args)
        LayerRunner(dev, log, hw).run(model, x0, expect, gap, fc, args.last_layer, args.first_layer)
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
        ok = LayerRunner(dev, log, hw, dump_dir).run(model, x0, expect, gap, fc, args.last_layer,
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
