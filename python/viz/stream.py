"""Per-sample (streaming) form of the frozen fully integer pipeline — the behaviour the firmware must have.

push(adc) consumes ONE ADC sample with bounded state and returns the events it completes:
  ("sample", idx, adc, filt)  for every sample
  ("beat", BeatResult)        when a beat is classified (needs the next detection and a complete window)
Bit-exact with the offline golden reference (scripts/export_c.py: filter_int, detect_offline, run_integer_beats,
quantise_rr_parts, int_forward_q) when started from zero state; tested on all golden segments.
"""

from collections import deque
from dataclasses import dataclass

import numpy as np

from python.dsp.pan_tompkins import DecisionEngine, FrontEnd, PTConfig
from python.quant.filter_int import quantize_sos
from python.quant.int8 import QuantModel, int_forward_q
from python.quant.pipeline_int import FILTER_FRAC_BITS, FILTER_GUARD_BITS, zscore_int

REFINE_HALF = 18


class IntegerFilterStream:
    """DF1 biquad cascade, Q20 coefficients, G guard bits: same arithmetic as sos_filter_int."""

    def __init__(self, sos: np.ndarray, frac_bits: int = FILTER_FRAC_BITS, guard_bits: int = FILTER_GUARD_BITS):
        self.coef = [tuple(int(v) for v in s) for s in quantize_sos(sos, frac_bits).tolist()]
        self.frac, self.guard = frac_bits, guard_bits
        self.half = 1 << (frac_bits - 1)
        self.state = [[0, 0, 0, 0] for _ in self.coef]          # x1, x2, y1, y2

    def push(self, x_int: int) -> int:
        v = int(x_int) << self.guard
        for (b0, b1, b2, _, a1, a2), st in zip(self.coef, self.state):
            x1, x2, y1, y2 = st
            y0 = (b0 * v + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2 + self.half) >> self.frac
            st[:] = [v, x1, y0, y1]
            v = y0
        return v


@dataclass(frozen=True)
class BeatResult:
    det_index: int               # index of the detection (sorted = emission order)
    r: int                       # refined R (integer abs refinement)
    r_coarse: int
    searchback: bool
    edge: bool
    emit_idx: int                # sample index at which the beat was classified
    z_q: np.ndarray              # (360,) int
    rr_parts: tuple              # rr_prev, rr_next, sum8, n8 (samples, clipped)
    logits: tuple                # 4 int (common scale)
    pred: int


class IntegerPipelineStream:
    def __init__(self, qm: QuantModel, pt: PTConfig, sos: np.ndarray, adc_zero: int, delay: int, half: int,
                 rr_lo: int, rr_hi: int, buffer: int = 4096):
        self.qm, self.pt = qm, pt
        self.adc_zero, self.delay, self.half = int(adc_zero), int(delay), int(half)
        self.rr_lo, self.rr_hi = int(rr_lo), int(rr_hi)
        self._fe = FrontEnd(pt, adc_zero)
        self._eng = DecisionEngine(pt)
        self._filt = IntegerFilterStream(sos)
        self._buf = [0] * buffer
        self._nbuf = buffer
        self.n = -1                                      # index of the last pushed sample
        self._dets = []                                  # (r_coarse, r_int, searchback, edge) of the last 2 detections
        self._n_det = 0
        self._rr = deque(maxlen=8)                       # past clipped RR (samples), for sum8 / n8
        self._pending = None                             # beat waiting for its window to complete
        self.max_latency = 0

    # ---- helpers
    def _y(self, i: int) -> int:
        if i <= self.n - self._nbuf or i > self.n:
            raise RuntimeError(f"sample {i} outside the {self._nbuf}-sample ring (n = {self.n})")
        return self._buf[i % self._nbuf]

    def _refine(self, p: int) -> tuple[int, bool]:
        c = p + self.delay
        lo, hi = max(c - REFINE_HALF, 0), c + REFINE_HALF
        if hi > self.n:
            raise RuntimeError("refinement window not yet available")
        best_k, best = 0, -1
        for k, i in enumerate(range(lo, hi + 1)):
            a = abs(self._y(i))
            if a > best:
                best_k, best = k, a
        if best_k == 0 or best_k == hi - lo:
            return p, True
        return lo + best_k - self.delay, False

    def _classify(self, beat: dict, out: list):
        c = beat["r"] + self.delay
        w = np.array([self._y(i) for i in range(c - self.half, c + self.half)], dtype=np.int64)
        z_q = zscore_int(w[None, :])
        from scripts.pipeline_int_eval import quantise_rr_parts
        parts = np.array([beat["parts"]], dtype=np.int64)
        logits = int_forward_q(self.qm, z_q, quantise_rr_parts(parts, self.qm.rr_exp, self.pt.fs))[0]
        self.max_latency = max(self.max_latency, self.n - beat["r"])
        out.append(("beat", BeatResult(beat["det_index"], beat["r"], beat["r_coarse"], beat["searchback"],
                                       beat["edge"], self.n, z_q[0], beat["parts"],
                                       tuple(int(v) for v in logits), int(np.argmax(logits)))))

    def _try_pending(self, out: list):
        b = self._pending
        if b is not None and b["r"] + self.delay + self.half - 1 <= self.n:
            self._pending = None
            self._classify(b, out)

    def _on_detection(self, det, out: list):
        if self._dets and det.r_coarse < self._dets[-1][0]:
            raise RuntimeError("detections not in increasing order")
        r, edge = self._refine(det.r_coarse)
        j = self._n_det
        self._n_det += 1
        if self._dets:
            rr = min(max(r - self._dets[-1][1], self.rr_lo), self.rr_hi)
            if j >= 2:
                # detection j-1 becomes a beat: rr_prev, rr_next, sum / count of the last <= 8 RR (incl. rr_prev)
                rr_prev = self._rr[-1]
                parts = (rr_prev, rr, sum(self._rr), len(self._rr))
                prev = self._dets[-1]
                if self._pending is not None:
                    raise RuntimeError("two beats pending")
                if prev[1] + self.delay - self.half >= 0:
                    self._pending = {"det_index": j - 1, "r": prev[1], "r_coarse": prev[0], "searchback": prev[2],
                                     "edge": prev[3], "parts": parts}
                    self._try_pending(out)
            self._rr.append(rr)
        self._dets = (self._dets + [(det.r_coarse, r, det.searchback, edge)])[-2:]

    # ---- public
    def push(self, adc: int) -> list:
        self.n += 1
        adc = int(adc)
        y = self._filt.push(adc - self.adc_zero)
        self._buf[self.n % self._nbuf] = y
        out = [("sample", self.n, adc, y)]
        self._try_pending(out)
        st = self._fe.push(adc)
        for det in self._eng.process(st[5], st[6], st[9]):
            self._on_detection(det, out)
        return out

    def flush(self) -> list:
        out = []
        for det in self._eng.flush():
            self._on_detection(det, out)
        self._pending = None                              # window can no longer complete
        return out
