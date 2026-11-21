"""Hardware survey of the FROZEN integer pipeline (read-only: it measures, it never changes a number).

Answers, for the RTL author: how wide must every register be, how many operations and how much memory does every
block need, and how many clock cycles are available. Two kinds of evidence are combined:
  * measured: maximum |value| of every stage over a signal (DS1 records or golden segments);
  * analytic: provable bounds from the L1 norm of the impulse responses and from sums of |weights|
    (the input is the MIT-BIH 11-bit ADC: |adc - adc_zero| <= 1024).
Hardware widths must use the analytic bound (a measured maximum on 22 records proves nothing about the next patient);
the measurement shows how conservative the bound is.
"""

from dataclasses import dataclass, field

import numpy as np

from python.dsp.pan_tompkins import DecisionEngine, PTConfig, frontend_offline
from python.quant.filter_int import quantize_sos, sos_filter_int
from python.quant.int8 import QuantModel, _conv_same, _maxpool2, rshift_round
from python.quant.pipeline_int import (
    FILTER_FRAC_BITS, FILTER_GUARD_BITS, quantise_rr_parts, round_div, run_integer_beats,
)

ADC_ABS_MAX = 1024                    # MIT-BIH: 11-bit ADC, zero at 1024 -> |x| <= 1024
STAGES = ("x", "s1", "lp", "hp_sum", "hp_scaled", "hp", "d", "q", "sq", "mwi")


def bits_signed(v) -> int:
    """Smallest two's-complement width that holds every value in [-|v|, |v|]."""
    return int(np.ceil(np.log2(abs(int(v)) + 1))) + 1


# ---------------------------------------------------------------- analytic bounds

def pt_bounds(cfg: PTConfig, x_max: int = ADC_ABS_MAX) -> dict[str, int]:
    """Worst-case |value| of every Pan-Tompkins stage, from the exact integer impulse responses."""
    ones = lambda n: np.ones(n, dtype=np.int64)               # noqa: E731
    h_s1 = ones(cfg.lp_len)
    h_lp = np.convolve(h_s1, ones(cfg.lp_len))
    hp_fir = -ones(cfg.hp_len)
    hp_fir[cfg.hp_delay] += cfg.hp_len                        # hp = hp_len * lp[n - delay] - sum(lp)
    h_hp = np.convolve(h_lp, hp_fir)
    h_d = np.convolve(h_hp, np.array([2, 1, 0, -1, -2], dtype=np.int64))
    out = {"x": x_max, "s1": int(np.abs(h_s1).sum()) * x_max, "lp": int(np.abs(h_lp).sum()) * x_max}
    out["hp_sum"] = int(np.abs(np.convolve(h_lp, ones(cfg.hp_len))).sum()) * x_max
    out["hp_scaled"] = cfg.hp_len * out["lp"]
    out["hp"] = int(np.abs(h_hp).sum()) * x_max
    out["d"] = int(np.abs(h_d).sum()) * x_max
    out["q"] = -(-out["d"] // (1 << cfg.deriv_shift))          # ceil(d / 2^shift)
    out["sq"] = out["q"] ** 2
    out["mwi"] = cfg.mwi_len * out["sq"]
    return out


def _impulse(b: np.ndarray, a: np.ndarray, n: int) -> np.ndarray:
    """Float impulse response of one biquad with the Q(F) coefficients (a0 = 1)."""
    h = np.zeros(n)
    x1 = x2 = y1 = y2 = 0.0
    for i in range(n):
        x0 = 1.0 if i == 0 else 0.0
        y0 = b[0] * x0 + b[1] * x1 + b[2] * x2 - a[1] * y1 - a[2] * y2
        h[i] = y0
        x2, x1, y2, y1 = x1, x0, y1, y0
    return h


def filter_bounds(sos: np.ndarray, frac_bits: int = FILTER_FRAC_BITS, guard_bits: int = FILTER_GUARD_BITS,
                  x_max: int = ADC_ABS_MAX, n_imp: int = 60000) -> dict:
    """Provable bounds of the 2-biquad DF1 filter: outputs of both sections and both accumulators."""
    sq = quantize_sos(sos, frac_bits)
    scale = 2.0 ** frac_bits
    sec = [(sq[i, :3] / scale, sq[i, 3:] / scale) for i in range(2)]
    h1, h2 = _impulse(*sec[0], n_imp), _impulse(*sec[1], n_imp)
    g1, g2 = _impulse(np.array([1.0, 0, 0]), sec[0][1], n_imp), _impulse(np.array([1.0, 0, 0]), sec[1][1], n_imp)
    h12 = np.convolve(h1, h2)[:n_imp]
    X = x_max << guard_bits
    # rounding adds at most 0.5 per sample at each accumulator output, filtered by the all-pole part (and by section 2)
    Y1 = np.abs(h1).sum() * X + 0.5 * np.abs(g1).sum()
    Y2 = np.abs(h12).sum() * X + 0.5 * np.abs(g1).sum() * np.abs(h2).sum() + 0.5 * np.abs(g2).sum()
    sb = np.abs(sq[:, :3]).sum(axis=1)
    sa = np.abs(sq[:, 4:]).sum(axis=1)
    acc1 = sb[0] * X + sa[0] * Y1
    acc2 = sb[1] * Y1 + sa[1] * Y2
    return {"x_in": X, "y1": int(np.ceil(Y1)), "y2": int(np.ceil(Y2)), "acc1": int(np.ceil(acc1)), "acc2": int(np.ceil(acc2)),
            "l1_h1": float(np.abs(h1).sum()), "l1_h12": float(np.abs(h12).sum()),
            "max_coef_bits": int(bits_signed(np.abs(sq).max())),
            "section2_numerator_shift_only": bool(np.array_equal(sq[1, :3], [1 << frac_bits, -2 << frac_bits, 1 << frac_bits])),
            "section1_b0_equals_b2": bool(sq[0, 0] == sq[0, 2])}


# ---------------------------------------------------------------- measurements on a signal

class OccupancyEngine(DecisionEngine):
    """DecisionEngine that also records the peak occupancy of its queues (read-only: same decisions)."""

    def __init__(self, cfg: PTConfig):
        super().__init__(cfg)
        self.max_sb = self.max_queue = 0

    def process(self, hp, d, mwi):
        out = super().process(hp, d, mwi)
        self.max_sb = max(self.max_sb, len(self._sb))
        self.max_queue = max(self.max_queue, len(self._queue))
        return out


@dataclass
class SignalSurvey:
    stages: dict = field(default_factory=dict)             # stage -> max |value|
    filter: dict = field(default_factory=dict)             # max_acc, max_y
    engine: dict = field(default_factory=dict)             # max_sb, max_queue, n_detections
    zscore: dict = field(default_factory=dict)             # bits of the z-score datapath
    cnn: dict = field(default_factory=dict)                # max |acc| per layer
    ring: dict = field(default_factory=dict)               # samples that must stay buffered
    n_beats: int = 0
    r_int: np.ndarray = None                                # refined R of every classified beat (test / analysis)
    emit: np.ndarray = None                                 # sample at which every beat is classified
    span: np.ndarray = None                                 # filtered samples that must stay buffered, per beat


def cnn_accumulators(qm: QuantModel, z_q: np.ndarray, rr_q: np.ndarray, batch: int = 256) -> dict[str, int]:
    """Maximum |accumulator| of every CNN layer for a set of beats (exactly the integers the int8 CNN adds up)."""
    mx: dict[str, int] = {}

    def upd(k, a):
        mx[k] = max(mx.get(k, 0), int(np.abs(a).max()))

    for i in range(0, len(z_q), batch):
        h = np.asarray(z_q[i:i + batch], dtype=np.int64)[:, None, :]
        for li, L in enumerate(qm.convs):
            acc = _conv_same(h, L["w_q"]) + L["b_q"][None, :, None]
            upd(f"conv{li}", acc)
            h = np.clip(rshift_round(acc, L["k"] + L["a_in"] - L["a_out"]), 0, 127)
            if L["pool"]:
                h = _maxpool2(h)
        s = h.sum(axis=2)
        f1, f2 = qm.fc1, qm.fc2
        acc_g = s @ f1["wg_q"].T
        acc_r = np.asarray(rr_q[i:i + batch], dtype=np.int64) @ f1["wr_q"].T
        acc1 = rshift_round(acc_g, f1["Eg"] - f1["E"]) + rshift_round(acc_r, f1["Er"] - f1["E"]) + f1["b_q"][None, :]
        hq = np.clip(rshift_round(acc1, f1["E"] - f1["a_out"]), 0, 127)
        upd("gap_sum", s)
        upd("fc1_gap", acc_g)
        upd("fc1_rr", acc_r)
        upd("fc1_total", acc1)
        upd("fc2_logit", hq @ f2["w_q"].T + f2["b_q"][None, :])
    return mx


def cnn_bounds(qm: QuantModel) -> dict[str, int]:
    """Provable |accumulator| bound per layer: every activation at its maximum (127), every weight at its worst sign."""
    out = {}
    for i, L in enumerate(qm.convs):
        out[f"conv{i}"] = int(np.abs(L["w_q"]).sum(axis=(1, 2)).max()) * 127 + int(np.abs(L["b_q"]).max())
    f1, f2 = qm.fc1, qm.fc2
    gap_len = 45
    out["gap_sum"] = gap_len * 127
    out["fc1_gap"] = int(np.abs(f1["wg_q"]).sum(axis=1).max()) * gap_len * 127
    out["fc1_rr"] = int(np.abs(f1["wr_q"]).sum(axis=1).max()) * 127
    out["fc1_total"] = (out["fc1_gap"] >> int((f1["Eg"] - f1["E"]).min())) + (out["fc1_rr"] >> int((f1["Er"] - f1["E"]).min())) \
        + int(np.abs(f1["b_q"]).max()) + 2
    out["fc2_logit"] = int(np.abs(f2["w_q"]).sum(axis=1).max()) * 127 + int(np.abs(f2["b_q"]).max())
    return out


def survey_signal(adc: np.ndarray, adc_zero: int, pt: PTConfig, sos: np.ndarray, delay: int, half: int, rr_lo: int,
                  rr_hi: int, qm: QuantModel | None = None) -> SignalSurvey:
    """Measure every stage over one signal (a DS1 record or a golden segment), starting from the zero state."""
    adc = np.asarray(adc, dtype=np.int64)
    s = SignalSurvey()
    st = frontend_offline(adc, adc_zero, pt)
    s.stages = {k: int(np.abs(st[k]).max()) for k in STAGES}
    sq = quantize_sos(sos, FILTER_FRAC_BITS)
    y_int, macc, my = sos_filter_int(adc - adc_zero, sq, FILTER_FRAC_BITS, FILTER_GUARD_BITS, track=True)
    s.filter = {"max_acc": int(macc), "max_y": int(my)}
    eng = OccupancyEngine(pt)
    dets = []
    for hp, d, mwi in zip(st["hp"].tolist(), st["d"].tolist(), st["mwi"].tolist()):
        dets += eng.process(hp, d, mwi)
    dets += eng.flush()
    dets.sort(key=lambda x: x.r_coarse)
    s.engine = {"max_sb": eng.max_sb, "max_queue": eng.max_queue, "n_detections": len(dets),
                "searchback": int(sum(d.searchback for d in dets))}
    if len(dets) < 3:
        return s
    ib = run_integer_beats(y_int, np.array([d.r_coarse for d in dets], dtype=np.int64), delay, half, rr_lo, rr_hi)
    bi, r_int, w = ib["beat_index"], ib["r_int"], ib["windows"]
    s.n_beats = len(bi)
    if not len(bi):
        return s
    n = w.shape[1]
    sm = w.sum(axis=1)
    m = round_div(sm, n)[:, None]
    dd = w - m
    sqs = (dd * dd).sum(axis=1)
    var = round_div(sqs, n)
    s.zscore = {"w": int(np.abs(w).max()), "sum": int(np.abs(sm).max()), "d": int(np.abs(dd).max()),
                "sum_d2": int(sqs.max()), "var": int(var.max()), "sd": int(np.sqrt(var.max())),
                "d_shift4": int(np.abs(dd << 4).max())}
    # buffering: samples of the filtered signal needed between the window start and the emission of the beat
    emit = np.array([max(dets[j + 1].emitted_at, r_int[j] + delay + half - 1) for j in bi], dtype=np.int64)
    span = emit - (r_int[bi] + delay - half) + 1
    s.ring = {"max_span": int(span.max()), "max_latency_r_to_emit": int((emit - r_int[bi]).max())}
    s.r_int, s.emit, s.span = r_int[bi], emit, span
    if qm is not None:
        s.cnn = cnn_accumulators(qm, ib["z_q"], quantise_rr_parts(ib["rr_parts"], qm.rr_exp))
    return s


def merge(a: dict, b: dict) -> dict:
    """Element-wise maximum of two dicts of non-negative numbers."""
    return {k: max(a.get(k, 0), b.get(k, 0)) for k in set(a) | set(b)}


def zscore_bounds(y_out_bound: int, n: int = 360) -> dict[str, int]:
    """Provable bounds of the integer z-score datapath, from the bound of the filtered window values."""
    d = 2 * y_out_bound                                     # d = w - round(mean): both within +-y_out_bound
    return {"w": y_out_bound, "sum": n * y_out_bound, "d": d, "sum_d2": n * d * d, "var": d * d, "sd": d,
            "d_shift4": d << 4}


def learning_queue_bound(cfg: PTConfig) -> int:
    """Peaks queued during the learning phase: MWI peaks are at least peak_confirm samples apart."""
    return cfg.learn_len // cfg.peak_confirm + 1


def ring_report(spans: np.ndarray, thresholds=(512, 1024, 1536, 2048, 3072, 4096)) -> dict:
    spans = np.asarray(spans)
    return {"beats": int(len(spans)), "max": int(spans.max()),
            "percentile": {str(q): int(np.percentile(spans, q)) for q in (50, 90, 99, 99.9)},
            "beats_over": {str(t): int((spans > t).sum()) for t in thresholds}}


# ---------------------------------------------------------------- static cost of the CNN

def cnn_static(qm: QuantModel) -> dict:
    lens = [360, 180, 90, 45]
    layers, macs = [], 0
    for i, (L, ln) in enumerate(zip(qm.convs, lens)):
        co, ci, k = L["w_q"].shape
        n = co * ci * k * ln
        macs += n
        layers.append({"name": f"conv{i}", "cout": co, "cin": ci, "k": k, "len": ln, "macs": n,
                       "weight_bytes": co * ci * k, "bias_bytes": 4 * co, "shift_bytes": co,
                       "out_bytes_before_pool": co * ln, "pool": bool(L["pool"])})
    f1, f2 = qm.fc1, qm.fc2
    fc1, fc2 = f1["wg_q"].size + f1["wr_q"].size, f2["w_q"].size
    macs += fc1 + fc2
    layers += [{"name": "fc1", "macs": fc1, "weight_bytes": fc1, "bias_bytes": 4 * f1["b_q"].size},
               {"name": "fc2", "macs": fc2, "weight_bytes": fc2, "bias_bytes": 4 * f2["b_q"].size}]
    return {"layers": layers, "total_macs": macs}
