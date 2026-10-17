"""Integer inference for ECGCNN: int8 weights/activations, int32 accumulators, power-of-2 scales.

GOLDEN REFERENCE FOR THE C PORT. Every integer operation here must be reproduced bit-exactly in C:

  requantize:  rshift_round(acc, s) = (acc + (1 << (s - 1))) >> s      (s > 0)
               with an ARITHMETIC right shift of a signed integer (floor division by 2^s).
               GCC implements >> on signed ints as an arithmetic shift; C must use exactly this
               expression (no float rounding, no round-half-to-even). For s == 0: acc; s < 0: acc << -s.
  ReLU + saturation: clamp(x, 0, 127).

Pipeline (per beat):
  x  : z-scored window (360, clipped +-8)  -> x_q = clamp(floor(x * 2^4 + 0.5), -127, 127)
  4 x conv (BN folded, int8 per-output-channel weights w_q = round(W' * 2^k_c), int32 bias
       b_q = round(b' * 2^(k_c + a_in))): acc = sum w_q * x_q + b_q;
       y_q = clamp(rshift_round(acc, k_c + a_in - a_out), 0, 127); MaxPool(2) after blocks 1-3
  GAP: S = sum over the 45 positions (int32, scale 2^-a4); the 1/45 is folded into FC1
  RR : rr_q[j] = clamp(floor(rr[j] * 2^r_j + 0.5), -127, 127) with a power-of-2 scale per feature;
       the z-score mu/sigma are folded into FC1 (W1r' = W1r / sigma, b1' = b1 - W1r @ (mu / sigma))
  FC1: two int32 partial sums per output row: GAP block (own weight exponent eg_o, accumulator
       exponent Eg_o = a4 + eg_o) and RR block (accumulator exponent Er_o, weights at Er_o - r_j).
       They are aligned to E_o = min(Eg_o, Er_o) with rshift_round, then the bias (at 2^E_o) is added:
       acc = rshift_round(acc_g, Eg_o - E_o) + rshift_round(acc_r, Er_o - E_o) + b_q
       h_q = clamp(rshift_round(acc, E_o - a5), 0, 127)
       (a single shared exponent would round the tiny GAP weights - already divided by 45 - to zero)
  FC2: per-tensor exponent -> int32 logits on one common scale -> argmax.
Weights/activation exponents are calibrated on float activations of a calibration set
(the fold's inner-train): max|x| (default) or a percentile.
"""

from dataclasses import dataclass, field

import numpy as np
import torch

from python.models.cnn1d import ECGCNN

Q = 127
INPUT_EXP = 4            # z-score input in [-8, 8] -> x * 16, saturated to +-127
INT32_MAX = 2**31 - 1


def rshift_round(acc: np.ndarray, s: int) -> np.ndarray:
    """(acc + (1 << (s - 1))) >> s on int64 with arithmetic shift. See module docstring."""
    acc = np.asarray(acc, dtype=np.int64)
    s = np.asarray(s, dtype=np.int64)
    if s.ndim == 0:
        s = int(s)
        if s > 0:
            return (acc + (np.int64(1) << np.int64(s - 1))) >> np.int64(s)
        return acc if s == 0 else acc << np.int64(-s)
    # per-channel shifts, broadcast over the channel axis (axis 1)
    out = np.empty_like(acc)
    for c, sc in enumerate(s.tolist()):
        out[:, c] = rshift_round(acc[:, c], sc)
    return out


def pow2_exp(maxabs: float, qmax: int = Q) -> int:
    """Largest integer k with maxabs * 2^k <= qmax."""
    if maxabs <= 0:
        return 0
    return int(np.floor(np.log2(qmax / maxabs)))


def quantize_input(x: np.ndarray, exp: int) -> np.ndarray:
    return np.clip(np.floor(np.asarray(x, dtype=np.float64) * 2.0**exp + 0.5), -Q, Q).astype(np.int64)


def _check_int32(acc: np.ndarray, where: str) -> None:
    m = int(np.abs(acc).max()) if acc.size else 0
    if m > INT32_MAX:
        raise OverflowError(f"{where}: |acc| = {m} exceeds int32")


# ---------------------------------------------------------------- float folding

def fold_bn(conv: torch.nn.Conv1d, bn: torch.nn.BatchNorm1d) -> tuple[np.ndarray, np.ndarray]:
    """Conv (no bias) + BN (eval) -> (W', b') with W' = W * g / sqrt(var + eps), b' = beta - g * mean / sqrt(var + eps)."""
    w = conv.weight.detach().double().cpu().numpy()
    g = bn.weight.detach().double().cpu().numpy()
    beta = bn.bias.detach().double().cpu().numpy()
    mean = bn.running_mean.detach().double().cpu().numpy()
    var = bn.running_var.detach().double().cpu().numpy()
    scale = g / np.sqrt(var + bn.eps)
    return w * scale[:, None, None], beta - mean * scale


@dataclass
class FoldedModel:
    """Float model with BN folded, 1/45 and RR mu/sigma folded into FC1 (float64)."""
    convs: list[tuple[np.ndarray, np.ndarray]]      # (W', b') per block
    pools: list[bool]
    w1_gap: np.ndarray                               # (H, C) already divided by the GAP length
    w1_rr: np.ndarray                                # (H, n_rr) already divided by sigma
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray
    gap_len: int


def fold_model(model: ECGCNN, in_len: int = 360) -> FoldedModel:
    model = model.eval().cpu()
    layers = list(model.features)
    convs, pools, i = [], [], 0
    while i < len(layers):
        conv, bn = layers[i], layers[i + 1]
        convs.append(fold_bn(conv, bn))
        i += 3                                       # Conv, BN, ReLU
        pool = i < len(layers) and isinstance(layers[i], torch.nn.MaxPool1d)
        pools.append(pool)
        if pool:
            i += 2                                   # MaxPool, Dropout (no-op at inference)
    gap_len = in_len // 2 ** sum(pools)
    w1 = model.fc1.weight.detach().double().numpy()
    b1 = model.fc1.bias.detach().double().numpy()
    c = convs[-1][0].shape[0]
    mu = model.rr_mean.double().numpy()
    sd = model.rr_std.double().numpy()
    w1_rr = w1[:, c:] / sd[None, :]
    return FoldedModel(convs, pools, w1[:, :c] / gap_len, w1_rr, b1 - w1[:, c:] @ (mu / sd),
                       model.fc2.weight.detach().double().numpy(), model.fc2.bias.detach().double().numpy(),
                       gap_len)


def _conv_same(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Cross-correlation with zero 'same' padding. x (B, Cin, L), w (Cout, Cin, K) -> (B, Cout, L).
    Works for float64 and int64 (exact integer arithmetic)."""
    k = w.shape[2]
    xp = np.pad(x, ((0, 0), (0, 0), (k // 2, k // 2)))
    win = np.lib.stride_tricks.sliding_window_view(xp, k, axis=2)        # (B, Cin, L, K)
    return np.einsum("bclk,ock->bol", win, w, optimize=True)


def _maxpool2(x: np.ndarray) -> np.ndarray:
    L = x.shape[2] // 2 * 2
    return np.maximum(x[:, :, 0:L:2], x[:, :, 1:L:2])


def float_forward(fm: FoldedModel, x: np.ndarray, rr: np.ndarray, collect: bool = False):
    """Float reference of the folded model. x (B, 360), rr (B, n_rr). Optionally returns activations."""
    h = np.asarray(x, dtype=np.float64)[:, None, :]
    acts = []
    for (w, b), pool in zip(fm.convs, fm.pools):
        h = np.maximum(_conv_same(h, w) + b[None, :, None], 0.0)
        acts.append(h)
        if pool:
            h = _maxpool2(h)
    s = h.sum(axis=2)                                                      # GAP sum (1/len folded)
    z = np.maximum(s @ fm.w1_gap.T + np.asarray(rr, dtype=np.float64) @ fm.w1_rr.T + fm.b1, 0.0)
    logits = z @ fm.w2.T + fm.b2
    return (logits, acts, z) if collect else logits


# ---------------------------------------------------------------- integer model

@dataclass
class QuantModel:
    convs: list[dict] = field(default_factory=list)   # w_q, b_q, k (per channel), a_in, a_out, pool
    rr_exp: np.ndarray = None                          # per RR feature
    fc1: dict = None                                   # wg_q, wr_q, b_q, A (per row), a_in_gap, a_out
    fc2: dict = None                                   # w_q, b_q, k (per tensor), a_in
    input_exp: int = INPUT_EXP
    calib: str = "max"


def _stat(v: np.ndarray, calib: str) -> float:
    a = np.abs(v).ravel()
    if calib == "max":
        return float(a.max())
    if calib.startswith("p"):
        return float(np.percentile(a, float(calib[1:])))
    raise ValueError(f"unknown calibration {calib!r}")


def quantize_model(fm: FoldedModel, x_cal: np.ndarray, rr_cal: np.ndarray, calib: str = "max") -> QuantModel:
    """Weights: per-output-channel power-of-2 int8. Activations: per-tensor power-of-2 int8 exponents
    from float activations on the calibration set (max|x| or percentile, e.g. 'p99.99')."""
    _, acts, z = float_forward(fm, x_cal, rr_cal, collect=True)
    qm = QuantModel(calib=calib)
    a_in = INPUT_EXP
    for (w, b), pool, act in zip(fm.convs, fm.pools, acts):
        k = np.array([pow2_exp(np.abs(w[c]).max()) for c in range(w.shape[0])], dtype=np.int64)
        w_q = np.clip(np.round(w * 2.0 ** k[:, None, None]), -Q, Q).astype(np.int64)
        b_q = np.round(b * 2.0 ** (k + a_in)).astype(np.int64)
        a_out = pow2_exp(_stat(act, calib))
        qm.convs.append({"w_q": w_q, "b_q": b_q, "k": k, "a_in": a_in, "a_out": a_out, "pool": pool})
        a_in = a_out
    a_gap = a_in
    qm.rr_exp = np.array([pow2_exp(np.abs(rr_cal[:, j]).max()) for j in range(rr_cal.shape[1])], dtype=np.int64)
    H = fm.w1_gap.shape[0]
    # GAP block: own per-row weight exponent -> accumulator exponent Eg = a_gap + eg
    eg = np.array([pow2_exp(np.abs(fm.w1_gap[o]).max()) for o in range(H)], dtype=np.int64)
    Eg = a_gap + eg
    # RR block: one accumulator exponent Er per row shared by the features (weights at Er - r_j)
    Er = np.empty(H, dtype=np.int64)
    for o in range(H):
        cand = [int(qm.rr_exp[j]) + pow2_exp(abs(fm.w1_rr[o, j]))
                for j in range(fm.w1_rr.shape[1]) if fm.w1_rr[o, j] != 0]          # zero weights do not constrain
        Er[o] = min(cand) if cand else Eg[o]
    E = np.minimum(Eg, Er)                    # common accumulator exponent; the finer block is shifted down
    wg_q = np.clip(np.round(fm.w1_gap * 2.0 ** eg[:, None]), -Q, Q).astype(np.int64)
    wr_q = np.clip(np.round(fm.w1_rr * 2.0 ** (Er[:, None] - qm.rr_exp[None, :])), -Q, Q).astype(np.int64)
    b1_q = np.round(fm.b1 * 2.0 ** E).astype(np.int64)
    a5 = pow2_exp(_stat(z, calib))
    qm.fc1 = {"wg_q": wg_q, "wr_q": wr_q, "b_q": b1_q, "Eg": Eg, "Er": Er, "E": E, "a_gap": a_gap, "a_out": a5}
    k2 = pow2_exp(np.abs(fm.w2).max())
    qm.fc2 = {"w_q": np.clip(np.round(fm.w2 * 2.0**k2), -Q, Q).astype(np.int64),
              "b_q": np.round(fm.b2 * 2.0 ** (k2 + a5)).astype(np.int64), "k": k2, "a_in": a5}
    return qm


def conv_int(x_q: np.ndarray, layer: dict) -> np.ndarray:
    """One integer conv block: int32 accumulate, per-channel requantize, ReLU, saturate to [0, 127]."""
    acc = _conv_same(x_q, layer["w_q"]) + layer["b_q"][None, :, None]
    _check_int32(acc, "conv")
    y = rshift_round(acc, layer["k"] + layer["a_in"] - layer["a_out"])
    return np.clip(y, 0, Q)


def quantize_rr(rr: np.ndarray, rr_exp: np.ndarray) -> np.ndarray:
    return np.clip(np.floor(np.asarray(rr, dtype=np.float64) * 2.0 ** rr_exp + 0.5), -Q, Q).astype(np.int64)


def int_forward(qm: QuantModel, x: np.ndarray, rr: np.ndarray, batch: int = 256) -> np.ndarray:
    """Integer inference from FLOAT pre-processing outputs (z-scored window, RR in seconds) -> int64 logits."""
    return int_forward_q(qm, quantize_input(x, qm.input_exp), quantize_rr(rr, qm.rr_exp), batch)


def int_forward_q(qm: QuantModel, x_q: np.ndarray, rr_q: np.ndarray, batch: int = 256) -> np.ndarray:
    """Integer inference from INTEGER inputs: x_q (B, 360) int8-range window (scale 2^input_exp),
    rr_q (B, n_rr) int8-range RR features (scale 2^rr_exp). Returns int64 logits (common scale, use argmax)."""
    x_q = np.asarray(x_q, dtype=np.int64)
    rr_all = np.asarray(rr_q, dtype=np.int64)
    out = []
    for i in range(0, len(x_q), batch):
        h = x_q[i:i + batch][:, None, :]
        for layer in qm.convs:
            h = conv_int(h, layer)
            if layer["pool"]:
                h = _maxpool2(h)
        s = h.sum(axis=2)                                                  # int32 GAP sum
        rr_q = rr_all[i:i + batch]
        f1 = qm.fc1
        acc_g = s @ f1["wg_q"].T                                           # scale 2^-Eg (per row)
        acc_r = rr_q @ f1["wr_q"].T                                        # scale 2^-Er (per row)
        _check_int32(acc_g, "fc1 gap")
        _check_int32(acc_r, "fc1 rr")
        acc1 = (rshift_round(acc_g, f1["Eg"] - f1["E"]) + rshift_round(acc_r, f1["Er"] - f1["E"])
                + f1["b_q"][None, :])                                      # aligned to 2^-E
        _check_int32(acc1, "fc1")
        hq = np.clip(rshift_round(acc1, f1["E"] - f1["a_out"]), 0, Q)
        f2 = qm.fc2
        acc2 = hq @ f2["w_q"].T + f2["b_q"][None, :]
        _check_int32(acc2, "fc2")
        out.append(acc2)
    return np.concatenate(out)


# ---------------------------------------------------------------- persistence (export / DS2 evaluation)

def save_quant_model(qm: QuantModel, path) -> None:
    arrays = {"rr_exp": qm.rr_exp, "input_exp": np.array(qm.input_exp), "calib": np.array(qm.calib),
              "n_convs": np.array(len(qm.convs))}
    for i, c in enumerate(qm.convs):
        for key in ("w_q", "b_q", "k"):
            arrays[f"conv{i}_{key}"] = c[key]
        for key in ("a_in", "a_out", "pool"):
            arrays[f"conv{i}_{key}"] = np.array(c[key])
    for key, v in qm.fc1.items():
        arrays[f"fc1_{key}"] = np.asarray(v)
    for key, v in qm.fc2.items():
        arrays[f"fc2_{key}"] = np.asarray(v)
    np.savez(path, **arrays)


def load_quant_model(path) -> QuantModel:
    z = np.load(path)
    qm = QuantModel(calib=str(z["calib"]), input_exp=int(z["input_exp"]), rr_exp=z["rr_exp"].astype(np.int64))
    for i in range(int(z["n_convs"])):
        qm.convs.append({"w_q": z[f"conv{i}_w_q"].astype(np.int64), "b_q": z[f"conv{i}_b_q"].astype(np.int64),
                         "k": z[f"conv{i}_k"].astype(np.int64), "a_in": int(z[f"conv{i}_a_in"]),
                         "a_out": int(z[f"conv{i}_a_out"]), "pool": bool(z[f"conv{i}_pool"])})
    qm.fc1 = {k[4:]: (z[k].astype(np.int64) if z[k].ndim else int(z[k])) for k in z.files if k.startswith("fc1_")}
    qm.fc2 = {k[4:]: (z[k].astype(np.int64) if z[k].ndim else int(z[k])) for k in z.files if k.startswith("fc2_")}
    return qm


def int_forward_trace(qm: QuantModel, x_q: np.ndarray, rr_q: np.ndarray) -> dict[str, np.ndarray]:
    """Same integer computation as int_forward_q, returning every intermediate tensor (golden vectors):
    conv{i} (after requantise/ReLU/saturate, before pooling), pool{i}, gap (int32 sum), fc1 (int8),
    logits (int32)."""
    h = np.asarray(x_q, dtype=np.int64)[:, None, :]
    tr = {"input": h[:, 0, :].copy()}
    for i, layer in enumerate(qm.convs):
        h = conv_int(h, layer)
        tr[f"conv{i}"] = h
        if layer["pool"]:
            h = _maxpool2(h)
            tr[f"pool{i}"] = h
    s = h.sum(axis=2)
    tr["gap"] = s
    rr_q = np.asarray(rr_q, dtype=np.int64)
    f1 = qm.fc1
    acc_g, acc_r = s @ f1["wg_q"].T, rr_q @ f1["wr_q"].T
    acc1 = rshift_round(acc_g, f1["Eg"] - f1["E"]) + rshift_round(acc_r, f1["Er"] - f1["E"]) + f1["b_q"][None, :]
    _check_int32(acc1, "fc1")
    tr["fc1"] = np.clip(rshift_round(acc1, f1["E"] - f1["a_out"]), 0, Q)
    tr["logits"] = tr["fc1"] @ qm.fc2["w_q"].T + qm.fc2["b_q"][None, :]
    _check_int32(tr["logits"], "fc2")
    return tr
