"""Fully integer pre-processing (golden reference for C): filter -> R refinement -> window -> z-score -> RR.

All arithmetic is on Python/NumPy integers; rounding divisions and shifts are defined here once:
  round_div(a, b)  = floor((2a + b) / (2b))   for b > 0   (round half up, also for negative a)
  rshift_round     = (acc + (1 << (s-1))) >> s           (python/quant/int8.py)
  isqrt            = floor(sqrt(n)), exact integer square root

1. Filter 0.5-40 Hz: DF1 biquads, coefficients Q20, 12 guard bits, int64 accumulator (python/quant/filter_int.py).
   x_int = adc - adc_zero; y_int is in ADC counts * 2^12.
2. R refinement (abs mode): argmax |y_int| in [R_coarse + delay - 18, R_coarse + delay + 18]; edge -> keep coarse.
3. Window: y_int[c - 180 : c + 180], c = R + delay (delay = 2).
4. z-score: m = round_div(sum w, 360); d = w - m; var = round_div(sum d^2, 360); sd = isqrt(var);
   z_q = clamp(round_div(d * 2^4, sd + EPS_Q), -127, 127)   (scale 2^4 = the CNN input scale, clip +-8 -> +-127)
   EPS_Q = 1: the float pipeline's eps = 1e-6 mV is 1e-6 * 200 * 2^12 = 0.8 integer units.
5. RR (samples, clipped to [72, 1080]) -> int8 features with the model's power-of-2 exponents r_j:
   rr_prev_q = round_div(rr_prev * 2^r0, 360); rr_next_q = round_div(rr_next * 2^r1, 360);
   avg_q = round_div(sum8 * 2^r2, 360 * n8); ratio_q = round_div(rr_prev * n8 * 2^r3, sum8)   (all clamped to +-127)
   (ratio = rr_prev / (sum8 / n8) computed exactly as a rational number).
"""

import math

import numpy as np

from python.data.beats import rr_clip_samples
from python.quant.filter_int import quantize_sos, sos_filter_int

FILTER_FRAC_BITS = 20
FILTER_GUARD_BITS = 12
INPUT_EXP = 4
EPS_Q = 1
Q = 127


def round_div(a, b):
    """floor((2a + b) / (2b)) for b > 0; works element-wise on int64 arrays and on Python ints."""
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a = np.asarray(a, dtype=np.int64)
        b = np.asarray(b, dtype=np.int64)
        if np.any(b <= 0):
            raise ValueError("round_div needs b > 0")
        return np.floor_divide(2 * a + b, 2 * b)
    if b <= 0:
        raise ValueError("round_div needs b > 0")
    return (2 * a + b) // (2 * b)


def isqrt_array(v: np.ndarray) -> np.ndarray:
    return np.array([math.isqrt(int(x)) for x in np.asarray(v).ravel()], dtype=np.int64).reshape(np.shape(v))


def filter_int(adc: np.ndarray, adc_zero: int, sos: np.ndarray) -> np.ndarray:
    sq = quantize_sos(sos, FILTER_FRAC_BITS)
    return sos_filter_int(np.asarray(adc, dtype=np.int64) - int(adc_zero), sq, FILTER_FRAC_BITS, FILTER_GUARD_BITS)


def refine_abs_int(r_coarse: np.ndarray, y_int: np.ndarray, delay: int, half: int) -> np.ndarray:
    out = np.empty(len(r_coarse), dtype=np.int64)
    a = np.abs(y_int)
    for i, p in enumerate(np.asarray(r_coarse, dtype=np.int64)):
        c = int(p) + delay
        lo, hi = max(c - half, 0), min(c + half, len(y_int) - 1)
        if hi <= lo:
            out[i] = p
            continue
        k = int(np.argmax(a[lo:hi + 1]))
        out[i] = p if (k == 0 or k == hi - lo) else lo + k - delay
    return out


def zscore_int(w: np.ndarray) -> np.ndarray:
    """w: (B, 360) int64 filtered windows -> (B, 360) int8-range z_q (scale 2^INPUT_EXP)."""
    w = np.asarray(w, dtype=np.int64)
    n = w.shape[1]
    m = round_div(w.sum(axis=1), n)[:, None]
    d = w - m
    var = round_div((d * d).sum(axis=1), n)
    sd = isqrt_array(var)[:, None]
    return np.clip(round_div(d << INPUT_EXP, sd + EPS_Q), -Q, Q)


def rr_int_features(r_pos: np.ndarray, rr_exp: np.ndarray, fs: int = 360, avg_n: int = 8,
                    rr_clip_s=(0.2, 3.0)) -> np.ndarray:
    """Quantised RR features for beats 1..n-2 of r_pos (same beat indexing as beats.rr_features)."""
    lo, hi = rr_clip_samples(rr_clip_s, fs)
    rr = np.clip(np.diff(np.asarray(r_pos, dtype=np.int64)), lo, hi)
    e = [int(v) for v in rr_exp]
    out = np.empty((max(len(r_pos) - 2, 0), 4), dtype=np.int64)
    for k, i in enumerate(range(1, len(r_pos) - 1)):
        past = rr[max(0, i - avg_n):i]
        s8, n8 = int(past.sum()), len(past)
        rp, rn = int(rr[i - 1]), int(rr[i])
        out[k] = (round_div(rp << e[0], fs), round_div(rn << e[1], fs),
                  round_div(s8 << e[2], fs * n8), round_div((rp * n8) << e[3], s8))
    return np.clip(out, -Q, Q)


def run_integer_beats(y_int: np.ndarray, r_coarse: np.ndarray, delay: int, half_win: int, rr_lo: int, rr_hi: int,
                      refine_half: int = 18) -> dict[str, np.ndarray]:
    """Integer pipeline after the filter, on a full R sequence (detector or annotation positions):
    integer abs refinement -> beats 1..n-2 with complete windows -> windows, z_q, RR parts
    (rr_prev, rr_next, sum of up to 8 past RR, count; samples, clipped). Shared by export and DS2 evaluation."""
    order = np.argsort(np.asarray(r_coarse), kind="stable")
    r_int = refine_abs_int(np.asarray(r_coarse)[order], y_int, delay, refine_half)
    rr = np.clip(np.diff(r_int), rr_lo, rr_hi)
    beats = [i for i in range(1, len(r_int) - 1)
             if r_int[i] + delay - half_win >= 0 and r_int[i] + delay + half_win <= len(y_int)]
    bi = np.array(beats, dtype=np.int64)
    if len(bi) == 0:
        return {"order": order, "r_int": r_int, "beat_index": bi, "windows": np.zeros((0, 2 * half_win), np.int64),
                "z_q": np.zeros((0, 2 * half_win), np.int64), "rr_parts": np.zeros((0, 4), np.int64)}
    windows = np.stack([y_int[r_int[i] + delay - half_win:r_int[i] + delay + half_win] for i in beats])
    parts = np.array([(rr[i - 1], rr[i], rr[max(0, i - 8):i].sum(), len(rr[max(0, i - 8):i])) for i in beats],
                     dtype=np.int64)
    return {"order": order, "r_int": r_int, "beat_index": bi, "windows": windows, "z_q": zscore_int(windows),
            "rr_parts": parts}
