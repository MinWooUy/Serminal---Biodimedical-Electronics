import math

import numpy as np

from python.data.beats import normalize_beats, rr_features
from python.dsp.filters import apply_offline, design_bandpass
from python.quant.int8 import quantize_input, quantize_rr
from python.quant.pipeline_int import (
    FILTER_GUARD_BITS, filter_int, isqrt_array, refine_abs_int, round_div, rr_int_features, zscore_int,
)


def test_round_div_half_up_including_negatives():
    for a in range(-50, 51):
        for b in (1, 2, 3, 7, 360):
            assert round_div(a, b) == math.floor(a / b + 0.5), (a, b)
    arr = np.arange(-50, 51)
    assert np.array_equal(round_div(arr, 7), np.floor(arr / 7 + 0.5).astype(np.int64))


def test_isqrt():
    v = np.array([0, 1, 2, 3, 4, 15, 16, 17, 10**12, 2**52 + 5])
    assert isqrt_array(v).tolist() == [math.isqrt(int(x)) for x in v]


def test_filter_int_close_to_float_and_stable():
    rng = np.random.default_rng(0)
    adc = (1024 + np.cumsum(rng.normal(0, 3, 20000))).astype(np.int64)
    sos = design_bandpass()
    y = filter_int(adc, 1024, sos) / 2.0**FILTER_GUARD_BITS / 200.0
    ref = apply_offline((adc - 1024) / 200.0, sos)
    assert np.sqrt(np.mean((y - ref) ** 2)) < 0.01 * np.sqrt(np.mean(ref**2))
    imp = np.zeros(20000, dtype=np.int64)
    imp[10] = 1000
    yi = filter_int(imp + 1024, 1024, sos)
    # Rounded DF1 with poles near 1 has a constant-output deadband instead of decaying to exactly 0:
    # a constant y survives while |y * (1 + a1 + a2)| < 1/2. Bound it per section and check the tail
    # is constant (no oscillation, no growth) and inside the bound (a few ADC LSB, i.e. microvolts).
    from python.quant.filter_int import quantize_sos
    sq = quantize_sos(sos, 20)
    bound = max(0.5 / ((sec[3] + sec[4] + sec[5]) / 2**20) for sec in sq)
    tail = yi[-2000:]
    assert np.ptp(tail) == 0                                 # settled to a constant
    assert abs(int(tail[0])) <= bound * 2                    # within the (cascaded) deadband
    assert abs(int(tail[0])) / 2.0**FILTER_GUARD_BITS < 4    # < 4 ADC LSB (= 20 uV at gain 200)


def test_zscore_int_matches_float_quantised():
    rng = np.random.default_rng(1)
    w = np.round(rng.normal(0, 2**12 * 50, size=(64, 360))).astype(np.int64)
    w[:, 180] += 2**12 * 600
    zi = zscore_int(w)
    zf = quantize_input(normalize_beats(w.astype(np.float64), 1e-6 * 200 * 2**12, 8.0), 4)
    assert np.abs(zi - zf).max() <= 1                        # only rounding differences
    assert np.mean(zi == zf) > 0.99


def test_rr_int_matches_float_quantised():
    rng = np.random.default_rng(2)
    r = np.cumsum(rng.integers(40, 1200, size=500))
    exp = np.array([5, 5, 6, 5])
    qi = rr_int_features(r, exp)
    qf = quantize_rr(rr_features(r, 360, rr_clip=(72, 1080)), exp)
    assert np.abs(qi - qf).max() <= 1
    assert np.mean(qi == qf) > 0.99


def test_refine_abs_int_edges():
    y = np.zeros(400, dtype=np.int64)
    y[202] = -500
    assert refine_abs_int(np.array([195]), y, 2, 18).tolist() == [200]
    y2 = np.arange(400, dtype=np.int64)                     # monotonic -> edge -> keep coarse
    assert refine_abs_int(np.array([150]), y2, 2, 18).tolist() == [150]


def test_run_integer_beats_consistent():
    from python.quant.pipeline_int import run_integer_beats
    rng = np.random.default_rng(3)
    y = np.round(rng.normal(0, 2**12 * 30, 20000)).astype(np.int64)
    r = np.sort(rng.choice(np.arange(400, 19600), size=40, replace=False))
    out = run_integer_beats(y, r[::-1], 2, 180, 72, 1080)                # unsorted input is sorted first
    assert np.all(np.diff(out["r_int"]) >= -36)                          # refinement moves by <= 18 each
    i = out["beat_index"]
    assert i.min() >= 1 and i.max() <= len(r) - 2
    for k, b in enumerate(i[:5]):
        c = out["r_int"][b] + 2
        assert np.array_equal(out["windows"][k], y[c - 180:c + 180])
    assert out["rr_parts"][:, 3].max() <= 8 and out["rr_parts"][:, :2].min() >= 72
