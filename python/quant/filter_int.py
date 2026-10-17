"""Fixed-point 0.5-40 Hz band-pass (Butterworth N=2 -> 2 biquads), Direct Form I. Golden reference for C.

Per section (coefficients in Q(F): c_q = round(c * 2^F), a0 = 1):
    acc  = b0q*x[n] + b1q*x[n-1] + b2q*x[n-2] - a1q*y[n-1] - a2q*y[n-2]
    y[n] = rshift_round(acc, F)          # (acc + (1 << (F-1))) >> F, arithmetic shift
Signals carry G guard bits: the first section gets x_int << G (x_int = adc - adc_zero), every
section output is in ADC units * 2^G. Output in mV: y / (2^G * adc_gain).
DF1 keeps only input/output histories (no internal state that can grow beyond the output range).
"""

import numpy as np

from python.quant.int8 import rshift_round


def quantize_sos(sos: np.ndarray, frac_bits: int) -> np.ndarray:
    """(n_sections, 6) float SOS -> int64 Q(F) coefficients (a0 kept as 2^F for reference)."""
    return np.round(np.asarray(sos, dtype=np.float64) * 2.0**frac_bits).astype(np.int64)


def pole_radii(sos_q: np.ndarray, frac_bits: int) -> np.ndarray:
    r = []
    for sec in sos_q:
        a = sec[3:] / 2.0**frac_bits
        r.extend(np.abs(np.roots(a)))
    return np.array(r)


def sos_filter_int(x_int: np.ndarray, sos_q: np.ndarray, frac_bits: int, guard_bits: int,
                   track: bool = False):
    """Integer DF1 cascade. x_int: int ADC counts (adc - adc_zero). Returns y (int64, ADC * 2^G);
    with track=True also (max |acc|, max |y|) over all sections."""
    y = (np.asarray(x_int, dtype=np.int64) << guard_bits).tolist()
    half = 1 << (frac_bits - 1)
    max_acc = max_y = 0
    for b0, b1, b2, _, a1, a2 in sos_q.tolist():
        out = [0] * len(y)
        x1 = x2 = y1 = y2 = 0
        for n, x0 in enumerate(y):
            acc = b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
            if track and abs(acc) > max_acc:
                max_acc = abs(acc)
            y0 = (acc + half) >> frac_bits             # rshift_round, arithmetic shift on Python ints
            out[n] = y0
            x2, x1, y2, y1 = x1, x0, y1, y0
        if track:
            max_y = max(max_y, max(abs(v) for v in out))
        y = out
    y = np.asarray(y, dtype=np.int64)
    return (y, max_acc, max_y) if track else y


assert int(rshift_round(np.array([-5]), 1)[0]) == ((-5 + 1) >> 1)   # same rounding rule as the CNN requantiser
