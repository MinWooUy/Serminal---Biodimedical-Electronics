"""Layer protocol of the FPGA team (layer_cmd_controller.v, CMD 08..15; RTL Serminal dev ca7562e): planner + command-level model.

The accelerator runs a WHOLE layer per RUN (bias + shift + ReLU [+ MaxPool(2)] [+ GAP]); FC layers are convs with Kw = 1
selected by CONFIG flags, so the PC only loads data and reads results back:

    layer_specs(qm)            -> the six layers of the frozen Lite CNN: conv0..conv3 (conv3 ends in GAP), fc1, fc2
    layer_operands(qm)         -> weights [Cout][K] (K = j*Cin + i), bias, shift (+ SG for fc1) per layer
    plan_layer(op, hw, ...)    -> commands of one layer: WRITE_W > SWAP > WRITE_BIAS > WRITE_SHIFT > CONFIG > RUN
    read_cmds(spec, hw)        -> READ_FMAP / READ_GAP commands that fetch the layer output
    run_beat(ops, z_q, rr_q, dev, hw, ...) -> logits, pred (+ every layer when read_layers) of one beat
    wire_cost(cmds, hw)        -> UART bytes / transactions / seconds (stop-and-wait, 8N1)
    LayerModel                 -> executes commands with the RTL's bank / flag / NACK rules (what the byte-level simulator wraps)

Only integer arithmetic. Everything that depends on the synthesized hardware (array size, SRAM depths, payload limit, clock,
baud) lives in HwConfig and is read from the board with INFO (0x15).
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field

import numpy as np

Q = 127
FRAME_OVERHEAD = 8                                   # A5 5A CMD SEQ LEN(2) ... CRC(2)
NACK_CRC, NACK_ARG, NACK_STATE, NACK_CMD = 1, 2, 3, 4
F_OUT_INT32, F_ARGMAX, F_ROW_COMBINE, F_GAP_IN = 0x1, 0x2, 0x4, 0x8
FLAG_NAMES = {F_OUT_INT32: "out_int32", F_ARGMAX: "argmax", F_ROW_COMBINE: "row_combine", F_GAP_IN: "gap_in"}
FEATURES_ALL = 0x1F                                  # bit 4: CONFIG w_base / b_base (resident weights)
BIAS_DEPTH = 256                                     # bias RAM entries (layer_cmd_controller.v BIAS_DEPTH; not in INFO)
INFO_FMT = "<BBBBIIHHHHH"                            # layer_cmd_controller.v header comment, 22 bytes
INFO_SIZE = struct.calcsize(INFO_FMT)
RUN_RESULT_BYTES = 5                                 # status u8 + cycles u32 LE
CYCLE_OVERHEAD_PER_TILE = 24                         # simulator only: NOT the RTL timing (the layer top does not model it here)


def flag_text(flags: int) -> str:
    return "+".join(v for k, v in FLAG_NAMES.items() if flags & k) or "none"


# ---------------------------------------------------------------- hardware description (INFO reply)
@dataclass(frozen=True)
class HwConfig:
    """Defaults = ecg_accel_top.v parameters. `INFO` returns the real ones."""
    rows: int = 3
    cols: int = 3
    clk_hz: int = 50_000_000
    baud: int = 921_600
    ifmap_depth: int = 4096
    wgt_depth: int = 16384
    feeder_depth: int = 1024
    cout_max: int = 64
    max_payload: int = 128
    features: int = FEATURES_ALL
    version: int = 2

    @classmethod
    def from_info(cls, p: bytes) -> "HwConfig":
        if len(p) != INFO_SIZE:
            raise ValueError(f"INFO reply is {len(p)} B, expected {INFO_SIZE} B")
        ver, rows, cols, feat, clk, baud, ifm, wgt, feed, cmax, mp = struct.unpack(INFO_FMT, p)
        return cls(rows, cols, clk, baud, ifm, wgt, feed, cmax, mp, feat if ver >= 2 else 0, ver)

    def to_info(self) -> bytes:
        return struct.pack(INFO_FMT, self.version, self.rows, self.cols, self.features, self.clk_hz, self.baud,
                           self.ifmap_depth, self.wgt_depth, self.feeder_depth, self.cout_max, self.max_payload)

    @property
    def write_chunk(self) -> int:                    # data bytes per WRITE_X / WRITE_W / WRITE_SHIFT (payload = addr u16 + data)
        return self.max_payload - 2

    @property
    def bias_chunk(self) -> int:                     # INT32 values per WRITE_BIAS (payload = addr u16 + 4 * n)
        return (self.max_payload - 2) // 4

    @property
    def read_chunk(self) -> int:                     # bytes per READ_FMAP (len is u8)
        return min(self.max_payload, 255)


# ---------------------------------------------------------------- commands
@dataclass(frozen=True)
class WriteX:
    base: int
    data: tuple


@dataclass(frozen=True)
class WriteW:
    base: int
    data: tuple


@dataclass(frozen=True)
class WriteBias:
    base: int
    values: tuple


@dataclass(frozen=True)
class WriteShift:
    base: int                                        # 0..COUT_MAX-1 = SO (requant), COUT_MAX.. = SG (row_combine pre-shift)
    data: tuple


@dataclass(frozen=True)
class Swap:
    x: bool
    w: bool


@dataclass(frozen=True)
class Config:
    out_len: int
    cout: int
    k: int
    step: int
    pool: int
    gap: int
    pad_next: int
    flags: int = 0
    w_base: int = 0                                  # start of the layer's weights in the READ weight bank
    b_base: int = 0                                  # start of the layer's biases in the bias RAM


@dataclass(frozen=True)
class Run:
    layer: str


@dataclass(frozen=True)
class ReadGap:
    pass


@dataclass(frozen=True)
class ReadFmap:
    offset: int
    length: int


@dataclass(frozen=True)
class Info:
    pass


def payload_len(cmd) -> int:
    if isinstance(cmd, (WriteX, WriteW, WriteShift)):
        return 2 + len(cmd.data)
    if isinstance(cmd, WriteBias):
        return 2 + 4 * len(cmd.values)
    if isinstance(cmd, Swap):
        return 1
    if isinstance(cmd, Config):
        return 12 if cmd.flags else 11
    if isinstance(cmd, ReadFmap):
        return 3
    return 0


def response_len(cmd, hw: HwConfig) -> int:
    """Whole answer frame in bytes."""
    if isinstance(cmd, Run):
        return FRAME_OVERHEAD + RUN_RESULT_BYTES
    if isinstance(cmd, ReadGap):
        return FRAME_OVERHEAD + 2 * hw.cout_max
    if isinstance(cmd, ReadFmap):
        return FRAME_OVERHEAD + cmd.length
    if isinstance(cmd, Info):
        return FRAME_OVERHEAD + INFO_SIZE
    return FRAME_OVERHEAD + 2


def wire_cost(cmds: list, hw: HwConfig) -> dict:
    """Bytes on the line for one command list (requests + answers), stop-and-wait, 10 bit times per byte."""
    tx = sum(FRAME_OVERHEAD + payload_len(c) for c in cmds)
    rx = sum(response_len(c, hw) for c in cmds)
    kinds: dict[str, int] = {}
    for c in cmds:
        kinds[type(c).__name__] = kinds.get(type(c).__name__, 0) + 1
    return {"transactions": len(cmds), "tx_bytes": tx, "rx_bytes": rx, "bytes": tx + rx,
            "seconds": (tx + rx) * 10 / hw.baud, "by_command": kinds}


# ---------------------------------------------------------------- RTL arithmetic
def post_quantize(acc: np.ndarray, s: np.ndarray) -> np.ndarray:
    """conv1d_post_process.v quantize() before ReLU / clamp; s per channel on the LAST axis, shifts >= 0 only
    (the frozen model has none below 3; a negative or >= 32 shift is rounded like the RTL's left shift / zero)."""
    acc = np.asarray(acc, dtype=np.int64)
    s = np.broadcast_to(np.asarray(s, dtype=np.int64), acc.shape)
    out = np.empty_like(acc)
    for v in np.unique(s):
        m = s == v
        a = acc[m]
        if v >= 32:
            out[m] = 0
        elif v > 0:
            out[m] = ((a >> (v - 1)) + 1) >> 1
        elif v == 0:
            out[m] = a
        elif v <= -8:
            out[m] = np.sign(a) * Q
        else:
            out[m] = a << (-v)
    return out


def rsr40(a: int, s: int) -> int:
    """rsr40() of conv1d_layer_top.v: (a + 2^(s-1)) >>> s; s == 0 or negative keeps a, s >= 32 gives 0."""
    if s == 0 or s < 0:
        return a
    if s >= 32:
        return 0
    return ((a >> (s - 1)) + 1) >> 1


# ---------------------------------------------------------------- layer description
@dataclass(frozen=True)
class LayerSpec:
    name: str
    cin: int                                         # channels per window position (FC1: GAP channels + RR bytes)
    kw: int
    cout: int
    out_len: int                                     # output positions before pooling / row_combine
    pool: bool
    gap: bool
    pad_next: int
    flags: int

    @property
    def k(self) -> int:
        return self.kw * self.cin

    @property
    def step(self) -> int:
        return self.cin

    @property
    def len_post(self) -> int:
        if self.flags & (F_ROW_COMBINE | F_OUT_INT32):
            return 1
        return self.out_len // 2 if self.pool else self.out_len

    @property
    def out_bytes(self) -> int:
        """Bytes of the layer output in the ifmap SRAM (0 for a GAP layer: its output is read with READ_GAP)."""
        if self.gap:
            return 0
        if self.flags & F_OUT_INT32:
            return 4 * self.cout + (1 if self.flags & F_ARGMAX else 0)
        return (self.len_post + 2 * self.pad_next) * self.cout


@dataclass
class LayerOperands:
    spec: LayerSpec
    weight: np.ndarray                               # int64, [Cout][K] flattened, K = j*Cin + i
    bias: np.ndarray
    shift: np.ndarray                                # SO
    preshift: np.ndarray | None = None               # SG (fc1 only)
    gap_cout: int = 0                                # fc1: GAP channels in front of the RR bytes
    aux_base: int = 0                                # fc1: ifmap address of the first RR byte (row 2, after the GAP part)


def layer_specs(qm, length: int = 360) -> list[LayerSpec]:
    specs, n = [], len(qm.convs)
    for i, c in enumerate(qm.convs):
        cout, cin, kw = (int(v) for v in np.shape(c["w_q"]))
        if kw % 2 == 0:
            raise ValueError(f"conv{i}: even kernel width {kw} has no symmetric padding")
        last = i == n - 1
        pad_next = 0 if last else (int(np.shape(qm.convs[i + 1]["w_q"])[2]) - 1) // 2
        pool = bool(c["pool"])
        specs.append(LayerSpec(f"conv{i}", cin, kw, cout, length, pool, last, pad_next, 0))
        if pool:
            length //= 2
    wg, wr = np.shape(qm.fc1["wg_q"]), np.shape(qm.fc1["wr_q"])
    if wg[1] != specs[-1].cout:
        raise ValueError(f"fc1 expects {wg[1]} GAP channels, conv{n - 1} has {specs[-1].cout}")
    w2 = np.shape(qm.fc2["w_q"])
    if w2[1] != wg[0]:
        raise ValueError(f"fc2 expects {w2[1]} inputs, fc1 has {wg[0]} outputs")
    specs.append(LayerSpec("fc1", wg[1] + wr[1], 1, wg[0], 3, False, False, 0, F_ROW_COMBINE | F_GAP_IN))
    specs.append(LayerSpec("fc2", w2[1], 1, w2[0], 1, False, False, 0, F_OUT_INT32 | F_ARGMAX))
    return specs


def _int8(a, what: str) -> np.ndarray:
    a = np.asarray(a, dtype=np.int64)
    if (a < -128).any() or (a > 127).any():
        raise ValueError(f"{what} does not fit int8")
    return a


def layer_operands(qm, length: int = 360) -> list[LayerOperands]:
    """Frozen QuantModel (params/model.npz) -> what the layer protocol loads. The shifts are the ones of weights.h."""
    specs = layer_specs(qm, length)
    ops = []
    for sp, c in zip(specs[:len(qm.convs)], qm.convs):
        w = _int8(c["w_q"], f"{sp.name} weights")                              # (cout, cin, kw)
        sh = np.asarray(c["k"], dtype=np.int64) + int(c["a_in"]) - int(c["a_out"])
        ops.append(LayerOperands(sp, w.transpose(0, 2, 1).reshape(sp.cout, -1).reshape(-1),
                                 np.asarray(c["b_q"], dtype=np.int64), _int8(np.broadcast_to(sh, (sp.cout,)), "shift")))
    f1, f2 = qm.fc1, qm.fc2
    sp = specs[len(qm.convs)]
    E, Eg, Er = (np.asarray(f1[k], dtype=np.int64) for k in ("E", "Eg", "Er"))
    if (Er != E).any():
        raise ValueError("fc1: Er != E, but row_combine adds the RR block without a shift (FC1_SHIFT_R must be 0)")
    gap_cout = int(np.shape(f1["wg_q"])[1])
    w1 = np.concatenate([_int8(f1["wg_q"], "fc1 GAP weights"), _int8(f1["wr_q"], "fc1 RR weights")], axis=1)
    ops.append(LayerOperands(sp, w1.reshape(-1), np.asarray(f1["b_q"], dtype=np.int64),
                             _int8(E - int(f1["a_out"]), "fc1 SO"), _int8(Eg - E, "fc1 SG"),
                             gap_cout, 2 * sp.step + gap_cout))
    sp = specs[-1]
    ops.append(LayerOperands(sp, _int8(f2["w_q"], "fc2 weights").reshape(-1), np.asarray(f2["b_q"], dtype=np.int64),
                             np.zeros(sp.cout, dtype=np.int64)))
    for op in ops:
        if (op.shift < 0).any() or (op.preshift is not None and (op.preshift < 0).any()):
            raise ValueError(f"{op.spec.name}: negative shift (the layer RTL only right-shifts)")
        if (np.abs(op.bias) > 2**31 - 1).any():
            raise ValueError(f"{op.spec.name}: bias does not fit int32")
    return ops


def fit_problems(sp: LayerSpec, hw: HwConfig) -> list[str]:
    """What this layer needs that the synthesized hardware cannot provide (empty list = fits)."""
    p = []
    if sp.cout > hw.cout_max:
        p.append(f"Cout {sp.cout} > COUT_MAX {hw.cout_max}")
    if sp.k > hw.feeder_depth:
        p.append(f"K {sp.k} > FEEDER_DEPTH {hw.feeder_depth}")
    if sp.cout * sp.k > hw.wgt_depth:
        p.append(f"weights {sp.cout * sp.k:,} B > weight SRAM {hw.wgt_depth:,} B")
    if not sp.gap and sp.out_bytes > hw.ifmap_depth:
        p.append(f"output {sp.out_bytes:,} B > ifmap SRAM {hw.ifmap_depth:,} B")
    if (sp.out_len - 1) * sp.step + sp.k > hw.ifmap_depth:
        p.append(f"input window {(sp.out_len - 1) * sp.step + sp.k:,} B > ifmap SRAM {hw.ifmap_depth:,} B")
    if sp.gap and 2 * hw.cout_max > hw.max_payload:
        p.append(f"READ_GAP answer {2 * hw.cout_max} B > payload limit {hw.max_payload} B")
    missing = sp.flags & ~hw.features
    if missing:
        p.append(f"needs CONFIG flags {flag_text(missing)}, the bitstream reports {flag_text(hw.features)} "
                 f"(INFO version {hw.version}): rebuild with the FC RTL")
    if sp.flags & F_ROW_COMBINE and hw.rows < 3:
        p.append(f"row_combine needs >= 3 PE rows, the array has {hw.rows}")
    return p


# ---------------------------------------------------------------- planner
def _writes(cls, flat, base: int, chunk: int) -> list:
    flat = [int(v) for v in flat]
    return [cls(base + o, tuple(flat[o:o + chunk])) for o in range(0, len(flat), chunk)]


def plan_layer(op: LayerOperands, hw: HwConfig, x_in=None, aux=None) -> list:
    """Commands of one layer. x_in: the input to load (first layer of a run only; later layers read the previous output
    from the ifmap SRAM). aux: the RR bytes of fc1 (written next to the on-chip GAP, gap_in packs the rest and swaps)."""
    sp = op.spec
    cmds: list = []
    if aux is not None:
        cmds += _writes(WriteX, aux, op.aux_base, hw.write_chunk)
    if x_in is not None:
        cmds += _writes(WriteX, x_in, 0, hw.write_chunk)
        cmds.append(Swap(True, False))
    cmds += _writes(WriteW, op.weight, 0, hw.write_chunk)
    cmds.append(Swap(False, True))
    bias = [int(v) for v in op.bias]
    cmds += [WriteBias(o, tuple(bias[o:o + hw.bias_chunk])) for o in range(0, len(bias), hw.bias_chunk)]
    cmds += _writes(WriteShift, op.shift, 0, hw.write_chunk)
    if op.preshift is not None:
        cmds += _writes(WriteShift, op.preshift, hw.cout_max, hw.write_chunk)
    cmds.append(Config(sp.out_len, sp.cout, sp.k, sp.step, int(sp.pool), int(sp.gap), sp.pad_next, sp.flags))
    cmds.append(Run(sp.name))
    return cmds


def read_cmds(sp: LayerSpec, hw: HwConfig) -> list:
    if sp.gap:
        return [ReadGap()]
    n, ch = sp.out_bytes, hw.read_chunk
    return [ReadFmap(o, min(ch, n - o)) for o in range(0, n, ch)]


def input_bytes(ops: list[LayerOperands], z_q) -> list[int]:
    """conv0 input: z_q with the symmetric zero padding of its kernel, channel-last X[t][c] (one channel)."""
    pad = (ops[0].spec.kw - 1) // 2
    return [0] * pad + [int(v) for v in z_q] + [0] * pad


def golden_input(ops: list[LayerOperands], g: dict, i: int, first: int) -> list[int]:
    """Input of layer `first` taken from the golden data of beat i (tests a layer alone, earlier errors cannot leak in)."""
    if first == 0:
        return input_bytes(ops, g["z_q"][i])
    sp = ops[first].spec
    if sp.name == "fc1":
        raise ValueError("fc1 cannot start from golden data: its GAP input is built on the chip by conv3")
    if sp.name == "fc2":
        return [int(v) for v in g["fc1"][i]]
    prev = ops[first - 1].spec
    key = f"pool{first - 1}" if prev.pool else f"conv{first - 1}"
    a = np.asarray(g[key][i], dtype=np.int64).T                                # (L, C)
    return np.pad(a, ((prev.pad_next, prev.pad_next), (0, 0))).reshape(-1).tolist()


def decode_output(sp: LayerSpec, data) -> dict:
    """Bytes read back from a layer -> arrays in golden shape; for fmap layers also whether the padding is all zero."""
    if sp.gap:
        return {"gap": np.asarray(data, dtype=np.int64)}
    b = np.frombuffer(bytes(data), dtype=np.uint8)
    if sp.flags & F_OUT_INT32:
        n = 4 * sp.cout
        logits = np.frombuffer(b[:n].tobytes(), dtype="<i4").astype(np.int64)
        return {"logits": logits, "pred": int(b[n]) if len(b) > n else -1}
    t = b.view(np.int8).astype(np.int64).reshape(-1, sp.cout)
    if sp.flags & F_ROW_COMBINE:                                               # fc1: one position, cout channels
        return {"h": t.reshape(-1)}
    pad = sp.pad_next
    inner = t[pad:len(t) - pad] if pad else t
    pad_ok = not (t[:pad].any() or t[len(t) - pad:].any()) if pad else True
    return {"fmap": inner.T, "pad_ok": pad_ok}


@dataclass
class BeatRun:
    logits: np.ndarray | None
    pred: int
    layers: dict                                     # name -> decoded output (see decode_output) of every layer read
    cmds: list
    status: dict                                     # layer -> RUN status byte (0 = OK, 1 = fmap writer overrun)
    cycles: dict                                     # layer -> RUN cycle count reported by the FPGA


def run_beat(ops: list[LayerOperands], z_q, rr_q, dev, hw: HwConfig, read_layers: bool = False, first: int = 0,
             last: int | None = None, golden: dict | None = None, beat: int = 0) -> BeatRun:
    """One beat through the layer accelerator (dev = anything with apply(cmd): LayerModel, LayerHost on real hardware).

    read_layers : read every layer's output (debug, ~9 KB more on the line); otherwise only the last layer is read.
    first / last: first and last layer index; with first > 0 the input comes from `golden` (layers.npz) beat `beat`.
    """
    last = len(ops) - 1 if last is None else last
    cmds_all: list = []
    out = BeatRun(None, -1, {}, cmds_all, {}, {})
    x_in = input_bytes(ops, z_q) if first == 0 else golden_input(ops, golden, beat, first)
    for n in range(first, last + 1):
        op = ops[n]
        sp = op.spec
        cmds = plan_layer(op, hw, x_in if n == first else None,
                          [int(v) for v in rr_q] if sp.name == "fc1" else None)
        if read_layers or n == last:
            cmds += read_cmds(sp, hw)
        data = bytearray()
        for c in cmds:
            r = dev.apply(c)
            if isinstance(c, Run):
                out.status[sp.name], out.cycles[sp.name] = r
            elif isinstance(c, ReadFmap):
                data += r
            elif isinstance(c, ReadGap):
                data = r
        cmds_all += cmds
        if read_layers or n == last:
            d = decode_output(sp, data)
            out.layers[sp.name] = d
            if "logits" in d:
                out.logits, out.pred = d["logits"], d["pred"]
    return out


def beat_cost(ops: list[LayerOperands], hw: HwConfig, read_layers: bool = False) -> dict:
    """UART cost of one beat (does not depend on the data)."""
    zeros = [0] * 360
    cmds: list = []
    for n, op in enumerate(ops):
        cmds += plan_layer(op, hw, input_bytes(ops, zeros) if n == 0 else None,
                           [0] * 4 if op.spec.name == "fc1" else None)
        if read_layers or n == len(ops) - 1:
            cmds += read_cmds(op.spec, hw)
    return wire_cost(cmds, hw)


# ---------------------------------------------------------------- command-level model (the RTL's rules)
class LayerNack(Exception):
    """The RTL would answer NACK with this code (layer_cmd_controller.v)."""

    def __init__(self, code: int, why: str):
        super().__init__(f"NACK {code}: {why}")
        self.code = code


class LayerUndefined(RuntimeError):
    """The host asked for something the RTL leaves undefined (reads outside the SRAM, int32 overflow, oversize answer)."""


@dataclass
class _Banks:
    depth: int
    data: np.ndarray = field(init=False)
    valid: np.ndarray = field(init=False)
    active: int = 0

    def __post_init__(self):
        self.data = np.zeros((2, self.depth), dtype=np.int64)
        self.valid = np.zeros((2, self.depth), dtype=bool)

    @property
    def stage(self) -> int:
        return 1 - self.active

    def write(self, base: int, values) -> None:
        values = np.asarray(values, dtype=np.int64)
        self.data[self.stage, base:base + len(values)] = values
        self.valid[self.stage, base:base + len(values)] = True

    def swap(self) -> None:
        self.active = 1 - self.active


class LayerModel:
    """Executes layer commands the way layer_cmd_controller + conv1d_layer_top do.

    strict: a read of a byte that was never written since reset raises LayerUndefined (host bug); otherwise it is counted in
    stats['unwritten_x' / 'unwritten_w']. Stale data in a bank is kept after a swap, as in the RTL (no valid map there).
    """

    def __init__(self, hw: HwConfig | None = None, strict: bool = False):
        self.hw = hw or HwConfig()
        self.strict = strict
        self.reset()

    def reset(self) -> None:
        hw = self.hw
        self.x, self.w = _Banks(hw.ifmap_depth), _Banks(hw.wgt_depth)
        self.bias = np.zeros(BIAS_DEPTH, dtype=np.int64)
        self.shift = np.zeros(hw.cout_max, dtype=np.int64)
        self.preshift = np.zeros(hw.cout_max, dtype=np.int64)
        self.gap_mem = np.zeros(hw.cout_max, dtype=np.int64)
        self.gap_cout = 0
        self.cfg: Config | None = None
        self.stats = {"unwritten_x": 0, "unwritten_w": 0, "layers": 0}

    # ------------------------------------------------------------ commands
    def apply(self, cmd):
        hw = self.hw
        if isinstance(cmd, WriteX):
            self._check_bytes(cmd.data)
            self._check_range(cmd.base, len(cmd.data), hw.ifmap_depth)
            self.x.write(cmd.base, cmd.data)
        elif isinstance(cmd, WriteW):
            self._check_bytes(cmd.data)
            self._check_range(cmd.base, len(cmd.data), hw.wgt_depth)
            self.w.write(cmd.base, cmd.data)
        elif isinstance(cmd, WriteShift):
            self._check_bytes(cmd.data)
            self._check_range(cmd.base, len(cmd.data), 2 * hw.cout_max)
            for i, v in enumerate(cmd.data):
                a = cmd.base + i
                (self.shift if a < hw.cout_max else self.preshift)[a % hw.cout_max] = v
        elif isinstance(cmd, WriteBias):
            self._check_range(cmd.base, len(cmd.values), BIAS_DEPTH)
            self.bias[cmd.base:cmd.base + len(cmd.values)] = cmd.values
        elif isinstance(cmd, Swap):
            if cmd.x:
                self.x.swap()
            if cmd.w:
                self.w.swap()
        elif isinstance(cmd, Config):
            self._config(cmd)
        elif isinstance(cmd, Run):
            return self._run()
        elif isinstance(cmd, ReadGap):
            return tuple(int(v) for v in self.gap_mem)
        elif isinstance(cmd, ReadFmap):
            if cmd.length == 0 or cmd.offset + cmd.length > hw.ifmap_depth:
                raise LayerNack(NACK_STATE, "READ_FMAP outside the ifmap SRAM")      # the RTL answers 3 here, not 2
            return bytes(int(v) & 0xFF for v in self.x.data[self.x.active, cmd.offset:cmd.offset + cmd.length])
        elif isinstance(cmd, Info):
            return hw.to_info()
        else:
            raise LayerNack(NACK_CMD, f"unknown command {cmd!r}")
        return None

    @staticmethod
    def _check_bytes(data) -> None:
        if any(not -128 <= int(v) <= 127 for v in data):
            raise ValueError("tensor bytes must be int8")

    @staticmethod
    def _check_range(base: int, n: int, limit: int) -> None:
        if n < 1 or base + n > limit:
            raise LayerNack(NACK_ARG, f"{n} B at {base} outside {limit}")

    def _config(self, c: Config) -> None:
        hw = self.hw
        f = c.flags
        i32, am, rc, gin = f & F_OUT_INT32, f & F_ARGMAX, f & F_ROW_COMBINE, f & F_GAP_IN
        bad_flags = (f >> 4 != 0 or (am and not i32) or (i32 and rc) or (gin and not rc)
                     or ((i32 or rc) and (c.pool != 0 or c.gap != 0 or c.pad_next != 0))
                     or (i32 and (c.out_len != 1 or 4 * c.cout + 1 > hw.ifmap_depth))
                     or (rc and (c.out_len != 3 or hw.rows < 3)))
        bad_base = c.w_base + c.cout * c.k > hw.wgt_depth or c.b_base + c.cout > BIAS_DEPTH
        if (bad_base or c.out_len == 0 or c.cout == 0 or c.cout > hw.cout_max or c.k == 0 or c.k > hw.feeder_depth or c.step == 0
                or c.pool > 1 or c.gap > 1 or (c.pool == 1 and c.out_len % 2) or bad_flags
                or not 0 <= c.pad_next <= 255):
            raise LayerNack(NACK_ARG, f"CONFIG rejected: {c}")
        self.cfg = c

    # ------------------------------------------------------------ one layer
    def _read_x(self, idx: np.ndarray) -> np.ndarray:
        if int(idx.max()) >= self.hw.ifmap_depth:
            raise LayerUndefined(f"layer reads ifmap address {int(idx.max())} >= {self.hw.ifmap_depth}")
        bank = self.x.active
        miss = int((~self.x.valid[bank, idx]).sum())
        if miss:
            self.stats["unwritten_x"] += miss
            if self.strict:
                raise LayerUndefined(f"layer reads {miss} ifmap bytes that were never written")
        return self.x.data[bank, idx]

    def _read_w(self, n: int, base: int = 0) -> np.ndarray:
        if base + n > self.hw.wgt_depth:
            raise LayerUndefined(f"layer needs {n} weight bytes at {base} > {self.hw.wgt_depth}")
        bank = self.w.active
        miss = int((~self.w.valid[bank, base:base + n]).sum())
        if miss:
            self.stats["unwritten_w"] += miss
            if self.strict:
                raise LayerUndefined(f"layer reads {miss} weight bytes that were never written")
        return self.w.data[bank, base:base + n]

    def _pack_gap(self, c: Config) -> None:
        """gap_in: gap_mem -> 3 rows (low 7 bits | high bits saturated to 127 | zeros) in the write bank, then swap."""
        hw = self.hw
        g = self.gap_mem.astype(np.int64)
        for r in range(3):
            for j in range(c.k):
                a = r * c.step + j
                inside = j < self.gap_cout
                if r == 2 and not inside:
                    continue                                                    # the host's bytes (RR) stay as written
                if a >= hw.ifmap_depth:
                    raise LayerUndefined("gap_in packs outside the ifmap SRAM")
                if not inside or r == 2:
                    v = 0
                elif r == 0:
                    v = int(g[j]) & 0x7F
                else:
                    v = 127 if (int(g[j]) >> 14) & 3 else (int(g[j]) >> 7) & 0x7F
                self.x.write(a, [v])
        self.x.swap()

    def _run(self):
        c = self.cfg
        if c is None:
            raise LayerNack(NACK_STATE, "RUN before CONFIG")
        hw = self.hw
        self.stats["layers"] += 1
        if c.gap:
            self.gap_mem[:] = 0
            self.gap_cout = c.cout
        if c.flags & F_GAP_IN:
            self._pack_gap(c)
        T, K = c.out_len, c.k
        idx = np.arange(T)[:, None] * c.step + np.arange(K)[None, :]
        X = self._read_x(idx)                                                  # (out_len, K)
        W = self._read_w(c.cout * K, c.w_base).reshape(c.cout, K)
        bias = self.bias[c.b_base:c.b_base + c.cout]
        raw = X @ W.T + bias[None, :]                            # every row is preloaded with the bias
        if (np.abs(raw) > 2**31 - 1).any():
            raise LayerUndefined("tile accumulator leaves int32")
        pad = 0 if c.gap else c.pad_next
        rows_tile = (2 if hw.rows >= 2 else 1) if c.pool else hw.rows
        tiles = math.ceil(T / rows_tile) * math.ceil(c.cout / hw.cols)
        cycles = tiles * (K + CYCLE_OVERHEAD_PER_TILE) + 1

        if c.flags & F_ROW_COMBINE:                                            # fc1: 3 PE rows of a channel -> one output
            h = np.empty(c.cout, dtype=np.int64)
            for ch in range(c.cout):
                t0, t1, t2 = (int(v) for v in raw[:, ch])
                b = int(bias[ch])
                comb_x = (t1 << 7) + t0 - (b << 7) - b                         # 128 * P1 + P0 (bias preloaded in every row)
                comb_y = rsr40(comb_x, int(self.preshift[ch])) + t2            # + P2 + bias
                h[ch] = min(max(rsr40(comb_y, int(self.shift[ch])), 0), Q)
            self._write_fmap(h, 0, c.cout, 1, 0)
        elif c.flags & F_OUT_INT32:                                            # fc2: raw int32 of row 0 + argmax
            r0 = raw[0]
            out = bytearray()
            for v in r0:
                out += struct.pack("<i", int(v))
            best = 0
            for ch in range(1, c.cout):                                        # strict '>' in channel order: ties keep the lowest
                if r0[ch] > r0[best]:
                    best = ch
            if c.flags & F_ARGMAX:
                out.append(best)
            self.x.write(0, [v - 256 if v > 127 else v for v in out])
            self.x.swap()
        else:
            y = np.clip(np.maximum(post_quantize(raw, self.shift[:c.cout][None, :]), 0), 0, Q)
            if c.pool:
                y = np.maximum(y[0::2], y[1::2])
            if c.gap:
                self.gap_mem[:c.cout] = (y.sum(axis=0)) & 0xFFFF
            else:
                self._write_fmap(y.reshape(-1), 0, c.cout, len(y), pad)
        return 0, cycles

    def _write_fmap(self, flat: np.ndarray, base: int, cout: int, n_pos: int, pad: int) -> None:
        """Zero padding rows at both ends, then the positions channel-last, then swap (auto_swap of the layer top)."""
        if pad:
            self.x.write(0, np.zeros(pad * cout, dtype=np.int64))
            self.x.write((n_pos + pad) * cout, np.zeros(pad * cout, dtype=np.int64))
        self.x.write(pad * cout + base, flat)
        self.x.swap()
