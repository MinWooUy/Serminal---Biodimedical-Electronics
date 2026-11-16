"""Run the frozen Lite int8 CNN on the FPGA team's tile accelerator (docs/proto_tile.md): planner + tile model.

The accelerator computes ONE tile per RUN_TILE: up to ROWS positions x COLS output channels of
    raw[r][c]  = (preload ? bias[c] : 0) + sum_k X[x_base + r*step + k] * W[w_base + c*K + k]
    post[r][c] = clamp(relu(round_shift(raw, shift[c])), -127, 127)
with two ping-pong byte banks (ifmap / weights, MEM_DEPTH bytes each, a valid bit per byte) and COLS bias registers.

    plan_beat(qm, z_q, rr_q)  -> list of commands for conv0..conv3, FC1 (RR part) and FC2 of one beat
    TileModel                 -> executes commands with the RTL's bank / validity / config rules (TileNack = RTL NACK)
    run_beat(qm, z_q, rr_q)   -> logits, pred and every intermediate layer; PC does MaxPool, GAP, the FC1 merge, argmax
    wire_cost(cmds)           -> UART bytes, transactions and seconds per beat (stop-and-wait, 8N1)

Only integer arithmetic; the PC-side parts use int8_ops.py (torch-free copies of python/quant/int8.py functions, checked
equal by tests), so the result must equal int_forward_q bit for bit.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

try:
    from python.link.int8_ops import Q, maxpool2, rshift_round
except ImportError:                       # stand-alone copy in the hand-off package
    from int8_ops import Q, maxpool2, rshift_round

# ---------------------------------------------------------------- RTL geometry (Serminal dev 4ad6c8b, de10_lite_top)
ROWS = 7
COLS = 7
MEM_DEPTH = 1024
FEEDER_DEPTH = 1024
MAX_PAYLOAD = 256
WRITE_CHUNK = MAX_PAYLOAD - 2          # bytes after the u16 base address
POST_ENABLE = True
FRAME_OVERHEAD = 8                     # A5 5A CMD SEQ LEN(2) ... CRC(2)
RESULT_BYTES = ROWS * COLS * 4 + (ROWS * COLS if POST_ENABLE else 0)
ACK_BYTES = FRAME_OVERHEAD + 2
STATUS_BYTES = FRAME_OVERHEAD + 3
NACK_LEN, NACK_CFG, NACK_NOT_READY, NACK_CMD = 2, 2, 3, 4


class TileNack(Exception):
    """The RTL would answer NACK with this code (docs/proto_tile.md section 4)."""

    def __init__(self, code: int, why: str):
        super().__init__(f"NACK {code}: {why}")
        self.code = code


# ---------------------------------------------------------------- commands
@dataclass(frozen=True)
class WriteIfmap:
    base: int
    data: tuple


@dataclass(frozen=True)
class WriteWeight:
    base: int
    data: tuple


@dataclass(frozen=True)
class WriteBias:
    base: int
    values: tuple


@dataclass(frozen=True)
class ConfigPost:
    shifts: tuple                      # COLS signed shifts, one per PE column
    relu: bool


@dataclass(frozen=True)
class ConfigTile:
    k: int
    step: int
    rows: int
    cols: int
    x_base: int
    w_base: int
    preload: bool


@dataclass(frozen=True)
class RunTile:
    layer: str                         # where the tile result goes
    t0: int
    c0: int


def payload_len(cmd) -> int:
    if isinstance(cmd, (WriteIfmap, WriteWeight)):
        return 2 + len(cmd.data)
    if isinstance(cmd, WriteBias):
        return 2 + 4 * len(cmd.values)
    if isinstance(cmd, ConfigPost):
        return COLS + 1
    if isinstance(cmd, ConfigTile):
        return 11 if cmd.preload else 10
    return 0


def response_len(cmd) -> int:
    return FRAME_OVERHEAD + RESULT_BYTES if isinstance(cmd, RunTile) else ACK_BYTES


def wire_cost(cmds: list, baud: int = 115200) -> dict:
    """Bytes on the line for one command list (requests + answers), stop-and-wait, 10 bit times per byte."""
    tx = sum(FRAME_OVERHEAD + payload_len(c) for c in cmds)
    rx = sum(response_len(c) for c in cmds)
    kinds: dict[str, int] = {}
    for c in cmds:
        kinds[type(c).__name__] = kinds.get(type(c).__name__, 0) + 1
    return {"transactions": len(cmds), "tiles": kinds.get("RunTile", 0), "tx_bytes": tx, "rx_bytes": rx,
            "bytes": tx + rx, "seconds": (tx + rx) * 10 / baud, "by_command": kinds}


# ---------------------------------------------------------------- tile model (bank lifecycle of host_command_controller.v)
def round_shift(acc: np.ndarray, s: int) -> np.ndarray:
    """conv1d_post_process.v quantize() before ReLU / clamp."""
    acc = np.asarray(acc, dtype=np.int64)
    if s >= 32:
        return np.zeros_like(acc)
    if s > 0:
        return ((acc >> (s - 1)) + 1) >> 1
    if s == 0:
        return acc
    if s <= -8:
        return np.sign(acc) * Q
    return acc << -s


@dataclass
class _Bank:
    data: np.ndarray = field(default_factory=lambda: np.zeros((2, MEM_DEPTH), dtype=np.int64))
    valid: np.ndarray = field(default_factory=lambda: np.zeros((2, MEM_DEPTH), dtype=bool))
    active: int = 0
    dirty: bool = False

    def write(self, base: int, values) -> None:
        if not values or base + len(values) > MEM_DEPTH:
            raise TileNack(NACK_LEN, f"write {len(values)} B at {base} outside {MEM_DEPTH} B")
        st = 1 - self.active
        self.data[st, base:base + len(values)] = values
        self.valid[st, base:base + len(values)] = True
        self.dirty = True

    def readable(self, idx: np.ndarray) -> bool:
        b = 1 - self.active if self.dirty else self.active
        return bool((idx < MEM_DEPTH).all()) and bool(self.valid[b, idx].all())

    def swap(self) -> None:
        if self.dirty:
            self.active = 1 - self.active
            self.valid[1 - self.active] = False
            self.dirty = False


class TileModel:
    """Command-level model of the accelerator; TileNack where the RTL answers NACK 2 / 3."""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.x, self.w = _Bank(), _Bank()
        self.bias = np.zeros(COLS, dtype=np.int64)
        self.bias_valid = np.zeros(COLS, dtype=bool)
        self.cfg: ConfigTile | None = None
        self.shifts: tuple | None = None
        self.relu = False

    def status(self) -> int:
        return (int(self.cfg is not None) | int(self.x.dirty) << 1 | int(self.w.dirty) << 2
                | int(bool(self.cfg and self.cfg.preload)) << 3 | int(self.shifts is not None) << 4)

    def apply(self, cmd):
        """Execute one command; returns (raw, post) arrays of shape (ROWS, COLS) for RunTile, else None."""
        if isinstance(cmd, WriteIfmap):
            self._check_bytes(cmd.data)
            self.x.write(cmd.base, cmd.data)
        elif isinstance(cmd, WriteWeight):
            self._check_bytes(cmd.data)
            self.w.write(cmd.base, cmd.data)
        elif isinstance(cmd, WriteBias):
            if not cmd.values or cmd.base + len(cmd.values) > COLS or len(cmd.values) * 4 + 2 > MAX_PAYLOAD:
                raise TileNack(NACK_LEN, "bias range")
            self.bias[cmd.base:cmd.base + len(cmd.values)] = cmd.values
            self.bias_valid[cmd.base:cmd.base + len(cmd.values)] = True
        elif isinstance(cmd, ConfigPost):
            if len(cmd.shifts) != COLS or any(not -128 <= s <= 127 for s in cmd.shifts):
                raise TileNack(NACK_LEN, "CONFIG_POST needs one int8 shift per column")
            self.shifts, self.relu = tuple(cmd.shifts), bool(cmd.relu)
        elif isinstance(cmd, ConfigTile):
            c = cmd
            if (not 0 < c.k <= FEEDER_DEPTH or c.step <= 0 or not 0 < c.rows <= ROWS or not 0 < c.cols <= COLS
                    or c.x_base + (c.rows - 1) * c.step + c.k > MEM_DEPTH or c.w_base + c.cols * c.k > MEM_DEPTH):
                raise TileNack(NACK_CFG, f"CONFIG_TILE rejected: {c}")
            self.cfg = c
        elif isinstance(cmd, RunTile):
            return self._run()
        else:
            raise TileNack(NACK_CMD, f"unknown command {cmd!r}")
        return None

    @staticmethod
    def _check_bytes(data) -> None:
        if any(not -128 <= v <= 127 for v in data):
            raise ValueError("tensor bytes must be int8")
        if len(data) > WRITE_CHUNK:
            raise TileNack(NACK_LEN, f"{len(data)} B > {WRITE_CHUNK} B per WRITE")

    def _run(self):
        c = self.cfg
        if c is None or (c.preload and not self.bias_valid[:c.cols].all()) or (POST_ENABLE and self.shifts is None):
            raise TileNack(NACK_NOT_READY, "not configured / bias or post missing")
        k = np.arange(c.k)
        xi = c.x_base + np.arange(c.rows)[:, None] * c.step + k[None, :]
        wi = c.w_base + np.arange(c.cols)[:, None] * c.k + k[None, :]
        if not self.x.readable(xi) or not self.w.readable(wi):
            raise TileNack(NACK_NOT_READY, "a needed ifmap / weight byte is not valid")
        self.x.swap()
        self.w.swap()
        X = self.x.data[self.x.active][xi]                      # (rows, K)
        W = self.w.data[self.w.active][wi]                      # (cols, K)
        raw = np.zeros((ROWS, COLS), dtype=np.int64)
        raw[:c.rows, :c.cols] = X @ W.T + (self.bias[:c.cols][None, :] if c.preload else 0)
        if (np.abs(raw) > 2**31 - 1).any():
            raise OverflowError("tile accumulator leaves int32")
        post = np.zeros((ROWS, COLS), dtype=np.int64)
        for j in range(c.cols):
            v = round_shift(raw[:c.rows, j], self.shifts[j])
            if self.relu:
                v = np.maximum(v, 0)
            post[:c.rows, j] = np.clip(v, -Q, Q)
        return raw, post


# ---------------------------------------------------------------- planner
@dataclass(frozen=True)
class LayerPlan:
    name: str
    cin: int
    k: int                 # kernel width (1 for FC)
    length: int            # output positions
    cout: int
    slab: int              # output channels whose weights share one weight-bank load
    seg: int               # output positions whose ifmap shares one ifmap-bank load
    order: str             # "seg_outer" or "slab_outer"


def _writes(cls, flat: np.ndarray) -> list:
    return [cls(o, tuple(int(v) for v in flat[o:o + WRITE_CHUNK])) for o in range(0, len(flat), WRITE_CHUNK)]


def max_slab(k: int, cin: int, cout: int) -> int:
    return min(cout, MEM_DEPTH // (k * cin))


def _layer_cmds(name: str, x_tc: np.ndarray, w_ck: np.ndarray, k: int, length: int, bias, shifts, relu: bool,
                order: str, slab: int, state: dict) -> tuple[list, LayerPlan]:
    """x_tc: (length + k - 1, cin) padded ifmap, time-major; w_ck: (cout, k * cin) weights in [k][ci] order."""
    cin = x_tc.shape[1]
    K = k * cin
    cout = w_ck.shape[0]
    if not 1 <= slab <= max_slab(k, cin, cout):
        raise ValueError(f"{name}: slab {slab} does not fit the weight bank")
    seg_max = MEM_DEPTH // cin - (k - 1)
    seg = min(length, seg_max if seg_max < ROWS else seg_max // ROWS * ROWS)
    if slab < 1 or seg < 1:
        raise ValueError(f"{name}: one window does not fit a {MEM_DEPTH} B bank")
    slabs = [(s0, min(slab, cout - s0)) for s0 in range(0, cout, slab)]
    segs = [(t0, min(seg, length - t0)) for t0 in range(0, length, seg)]
    pairs = [(sl, sg) for sg in segs for sl in slabs] if order == "seg_outer" else [(sl, sg) for sl in slabs for sg in segs]
    cmds = []
    for (s0, sn), (t0, tn) in pairs:
        if state.get("w") != (name, s0, sn):
            cmds += _writes(WriteWeight, w_ck[s0:s0 + sn].reshape(-1))
            state["w"] = (name, s0, sn)
        if state.get("x") != (name, t0, tn):
            cmds += _writes(WriteIfmap, x_tc[t0:t0 + tn + k - 1].reshape(-1))
            state["x"] = (name, t0, tn)
        for c0 in range(s0, s0 + sn, COLS):
            cols = min(COLS, s0 + sn - c0)
            if bias is not None:
                bv = tuple(int(v) for v in bias[c0:c0 + cols])
                if state.get("bias", (None,) * COLS)[:cols] != bv:
                    cmds.append(WriteBias(0, bv))
                    state["bias"] = bv + tuple(state.get("bias", (None,) * COLS)[cols:])
            sh = tuple(int(v) for v in shifts[c0:c0 + cols]) + (0,) * (COLS - cols) if shifts is not None else None
            post = (sh, relu) if sh is not None else state.get("post") or ((0,) * COLS, False)
            if state.get("post") != post:
                cmds.append(ConfigPost(*post))
                state["post"] = post
            for r0 in range(t0, t0 + tn, ROWS):
                rows = min(ROWS, t0 + tn - r0)
                cmds.append(ConfigTile(K, cin, rows, cols, (r0 - t0) * cin, (c0 - s0) * K, bias is not None))
                cmds.append(RunTile(name, r0, c0))
    return cmds, LayerPlan(name, cin, k, length, cout, slab, seg, order)


_CHOICE: dict = {}


def _choose(name, x_tc, w_ck, k, length, bias, shifts, relu) -> tuple[str, int]:
    """Order and slab size with the fewest UART bytes; the counts do not depend on the ifmap values (zeros used)."""
    key = (name, x_tc.shape, w_ck.shape, k, length)
    if key not in _CHOICE:
        zeros = np.zeros_like(x_tc)
        cands = []
        for slab in range(1, max_slab(k, x_tc.shape[1], w_ck.shape[0]) + 1):
            for order in ("seg_outer", "slab_outer"):
                cmds, _ = _layer_cmds(name, zeros, w_ck, k, length, bias, shifts, relu, order, slab, {})
                cands.append((wire_cost(cmds)["bytes"], order, slab))
        _CHOICE[key] = min(cands)[1:]
    return _CHOICE[key]


def _best_layer(name, x_tc, w_ck, k, length, bias, shifts, relu, state) -> tuple[list, LayerPlan]:
    order, slab = _choose(name, x_tc, w_ck, k, length, bias, shifts, relu)
    return _layer_cmds(name, x_tc, w_ck, k, length, bias, shifts, relu, order, slab, state)


def conv_operands(x_q: np.ndarray, layer: dict) -> tuple[np.ndarray, np.ndarray]:
    """(cin, L) activations and a conv layer of int8.py -> padded time-major ifmap and [k][ci] weights."""
    w = np.asarray(layer["w_q"], dtype=np.int64)                         # (cout, cin, k)
    k = w.shape[2]
    pad = (k - 1) // 2
    x_tc = np.pad(np.asarray(x_q, dtype=np.int64).T, ((pad, k - 1 - pad), (0, 0)))
    return x_tc, w.transpose(0, 2, 1).reshape(w.shape[0], -1)


def conv_shifts(layer: dict) -> np.ndarray:
    return np.asarray(layer["k"] + layer["a_in"] - layer["a_out"], dtype=np.int64)


def _collect(model: TileModel, cmds: list, outs: dict) -> None:
    for c in cmds:
        res = model.apply(c)
        if res is None:
            continue
        raw, post = res
        cfg = model.cfg
        o = outs[c.layer]
        o["raw"][c.c0:c.c0 + cfg.cols, c.t0:c.t0 + cfg.rows] = raw[:cfg.rows, :cfg.cols].T
        o["post"][c.c0:c.c0 + cfg.cols, c.t0:c.t0 + cfg.rows] = post[:cfg.rows, :cfg.cols].T


@dataclass
class BeatRun:
    logits: np.ndarray
    pred: int
    layers: dict           # conv0..conv3 (post), pool0..pool2, gap, fc1_rr (raw), fc1, logits
    cmds: list
    plans: list
    spans: dict            # layer -> (first, end) index into cmds


def run_beat(qm, z_q: np.ndarray, rr_q: np.ndarray, model=None, state: dict | None = None,
             upto: str | None = None) -> BeatRun:
    """One beat through the tile accelerator (model) with the PC doing MaxPool, GAP, the FC1 merge and argmax.

    The command list is planned layer by layer from the activations the tiles return, exactly as a host would.
    model: anything with apply(cmd) -> None | (raw, post) and a .cfg attribute: TileModel, or a TileHost on real hardware.
    state: what the banks / bias / post registers hold (carried across beats so unchanged weights are not re-sent).
    upto : stop after this layer (e.g. "conv0"); logits = None, pred = -1.
    """
    model = model or TileModel()
    state = {} if state is None else state
    state.pop("x", None)                                                       # ifmap bytes belong to the previous beat
    layers, all_cmds, plans, spans = {}, [], [], {}
    h = np.asarray(z_q, dtype=np.int64)[None, :]                              # (1, 360)
    for i, layer in enumerate(qm.convs):
        x_tc, w_ck = conv_operands(h, layer)
        cout, length = w_ck.shape[0], h.shape[1]
        cmds, plan = _best_layer(f"conv{i}", x_tc, w_ck, layer["w_q"].shape[2], length,
                                 layer["b_q"], conv_shifts(layer), True, state)
        outs = {plan.name: {"raw": np.zeros((cout, length), np.int64), "post": np.zeros((cout, length), np.int64)}}
        _collect(model, cmds, outs)
        h = outs[plan.name]["post"]
        layers[plan.name] = h
        spans[plan.name] = (len(all_cmds), len(all_cmds) + len(cmds))
        all_cmds += cmds
        plans.append(plan)
        if layer["pool"]:                                                      # PC: MaxPool(2)
            h = maxpool2(h[None])[0]
            layers[f"pool{i}"] = h
        if upto in (plan.name, f"pool{i}"):
            return BeatRun(None, -1, layers, all_cmds, plans, spans)
    s = h.sum(axis=1)                                                          # PC: GAP sum
    layers["gap"] = s
    f1, f2 = qm.fc1, qm.fc2
    # FPGA: FC1 RR part (raw int32, no bias, no requantisation)
    rr = np.asarray(rr_q, dtype=np.int64)
    cmds, plan = _best_layer("fc1_rr", rr[None, :], np.asarray(f1["wr_q"], np.int64), 1, 1, None, None, False, state)
    outs = {"fc1_rr": {"raw": np.zeros((f1["wr_q"].shape[0], 1), np.int64), "post": np.zeros((f1["wr_q"].shape[0], 1), np.int64)}}
    _collect(model, cmds, outs)
    acc_r = outs["fc1_rr"]["raw"][:, 0]
    layers["fc1_rr"] = acc_r
    spans[plan.name] = (len(all_cmds), len(all_cmds) + len(cmds))
    all_cmds += cmds
    plans.append(plan)
    # PC: FC1 GAP part + merge of the three shifts + bias + ReLU, exactly as int8.int_forward_q
    acc_g = s[None, :] @ np.asarray(f1["wg_q"], np.int64).T
    acc1 = (rshift_round(acc_g, f1["Eg"] - f1["E"]) + rshift_round(acc_r[None, :], f1["Er"] - f1["E"])
            + np.asarray(f1["b_q"], np.int64)[None, :])
    hq = np.clip(rshift_round(acc1, f1["E"] - f1["a_out"]), 0, Q)[0]
    layers["fc1"] = hq
    # FPGA: FC2 with bias preload, raw = logits
    cmds, plan = _best_layer("fc2", hq[None, :], np.asarray(f2["w_q"], np.int64), 1, 1, f2["b_q"], None, False, state)
    outs = {"fc2": {"raw": np.zeros((f2["w_q"].shape[0], 1), np.int64), "post": np.zeros((f2["w_q"].shape[0], 1), np.int64)}}
    _collect(model, cmds, outs)
    logits = outs["fc2"]["raw"][:, 0]
    layers["logits"] = logits
    spans[plan.name] = (len(all_cmds), len(all_cmds) + len(cmds))
    all_cmds += cmds
    plans.append(plan)
    pred = int(np.argmax(logits))                                             # first maximum = strict '>' scan
    return BeatRun(logits, pred, layers, all_cmds, plans, spans)
