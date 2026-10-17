"""Integer Pan-Tompkins QRS detector at 360 Hz (streaming + bit-identical offline).

Input: raw ADC counts; the detector works on x = adc - adc_zero (|x| <= 2^10 for
MIT-BIH 11-bit ADC). Every stage is integer-only and portable to C int32.

Conversion 200 Hz (Pan & Tompkins 1985) -> 360 Hz: every length/time is scaled
by 360/200 = 1.8 and rounded (HP length chosen odd so its delay is an integer).

  stage        200 Hz original             360 Hz here                         delay (samples)
  low-pass     (1-z^-6)^2/(1-z^-1)^2       2 cascaded running sums, len 11     10
  high-pass    z^-16 - MA32                57*lp[n-28] - sum57(lp)  (no div)   28
  derivative   (2,1,0,-1,-2)/8             2x[n]+x[n-1]-x[n-3]-2x[n-4]         2
  squaring     y^2                         (d >> 12)^2                         0
  MWI          30 samples (150 ms)         running sum len 54 (no div)         26.5
  refractory   200 ms                      72
  T-wave       360 ms                      130
  learning     2 s                         720

The derivative keeps 5 taps: at 360 Hz it is still a differentiator over the
QRS band, only its gain scale changes, which the adaptive thresholds absorb.
Measured 5-15 Hz band (LP+HP): -3 dB at 4.9-11.8 Hz, peak 8.0 Hz, as the original.

Total delays: band-pass output (LP+HP) = 38 samples; to the MWI output
38 + 2 + 26.5 ~= 66.5 samples. The coarse R position is taken from the
band-pass channel: r_coarse = bp_idx - 38 (raw-signal time).

int32 budget (worst case = L1 norm of the impulse response x 2^10):
  LP 2^16.9, HP 2^23.1, derivative 2^23.8, after >>12 2^11.8,
  square 2^23.5, MWI sum54 2^29.3 < 2^31.

Two threshold sets (as in the paper): I on the MWI output, F on |band-pass|.
All updates are integer with power-of-2 coefficients:
  SPK = (peak + 7*SPK) >> 3      implemented as SPK + ((peak - SPK) >> 3)
  NPK = (peak + 7*NPK) >> 3      (same rewrite)
  search-back: SPK = (peak + 3*SPK) >> 2  ->  SPK + ((peak - SPK) >> 2)
  THR1 = NPK + ((SPK - NPK) >> 2);  THR2 = THR1 >> 1
The rewritten forms are bit-identical (floor((8S + p - S)/8) = S + floor((p - S)/8))
and avoid the 7*SPK overflow in int32.

Learning phase: thresholds are initialised from the 2 s that start once the
front-end FIR stages have settled (settle = 133 samples):
  SPK0 = max >> 2,  NPK0 = (sum // 720) >> 1   (separately for I and F).
Peaks confirmed during learning are queued and classified right after.

MWI peak: a strict local maximum that stays the largest for 36 samples (0.1 s).

Pairing MWI peak p -> band-pass peak: the band-pass peak is argmax |hp| over
hp indices [p - 55, p - 2] (54 samples). Those are exactly the samples whose
derivative (delay 2) fed the 54-sample MWI sum at p. Slope for T-wave
discrimination = max |d| over d indices [p - 53, p].

RR handling (integer):
  RR_avg1 = mean of last 8 RR (sum >> 3); RR_avg2 = mean of last 8 RR within
  [0.921875, 1.15625] x RR_avg2 (shift-add for 92% / 116%).
  search-back limit RR_missed = RR_avg2 * 1.65625 (1 + 1/2 + 1/8 + 1/32 ~ 1.66).
  Before the first RR: RR_avg = 360. The first RR fills both 8-slot buffers.
  Addition to the paper (documented): if 8 consecutive RR fall outside the
  RR_avg2 limits, RR_avg2's buffer is reset to RR_avg1's (avoids lock-up after
  a rhythm change).
  irregular_halving (config): if the last RR was outside the limits, THR1 of
  both sets is halved, as in the paper.

Search-back fires once the gap since the last beat exceeds RR_missed plus the
detection latency (36 + 55 samples): the highest buffered noise peak above THR2
in both channels (refractory and T-wave rules applied) is declared a beat.
"""

from dataclasses import asdict, dataclass, fields

import numpy as np

from python.dsp.filters import apply_offline, design_bandpass, group_delay_compensation
from python.dsp.realign import half_window_samples, realign_one

STAGE_NAMES = ("x", "s1", "lp", "hp_sum", "hp_scaled", "hp", "d", "q", "sq", "mwi")


@dataclass(frozen=True)
class PTConfig:
    fs: int = 360
    lp_len: int = 11            # 6 @200 Hz x 1.8
    hp_len: int = 57            # 32 @200 Hz x 1.8, odd -> integer delay 28
    deriv_shift: int = 12       # right shift before squaring (int32 budget)
    mwi_len: int = 54           # 150 ms
    refractory: int = 72        # 200 ms
    twave_window: int = 130     # 360 ms
    learn_len: int = 720        # 2 s
    peak_confirm: int = 36      # 0.1 s
    sb_buffer: int = 16         # search-back ring buffer (peaks)
    hist_len: int = 128         # history ring for |hp|, |d| (power of 2)
    filt_buffer: int = 4096     # streaming ring of 0.5-40 Hz samples for R refinement
    irregular_halving: bool = True
    realign_mode: str = "abs"   # chosen on DS1 by scripts/align_stats.py

    @classmethod
    def from_dict(cls, d: dict) -> "PTConfig":
        names = {f.name for f in fields(cls)}
        unknown = set(d) - names
        if unknown:
            raise ValueError(f"unknown PT config keys: {sorted(unknown)}")
        return cls(**d)

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def lp_delay(self) -> int:
        return self.lp_len - 1              # two running sums, (L-1)/2 each

    @property
    def hp_delay(self) -> int:
        return (self.hp_len - 1) // 2

    @property
    def deriv_delay(self) -> int:
        return 2

    @property
    def bp_delay(self) -> int:
        return self.lp_delay + self.hp_delay

    @property
    def pair_back(self) -> int:
        return self.mwi_len + 1             # pairing window starts at p - pair_back

    @property
    def settle(self) -> int:
        # FIR length of LP+HP+derivative+MWI minus one: outputs before this see the zero history.
        return 2 * (self.lp_len - 1) + (self.hp_len - 1) + 4 + (self.mwi_len - 1)


@dataclass
class Detection:
    bp_idx: int          # band-pass (hp) peak index
    mwi_idx: int         # MWI peak index
    r_coarse: int        # bp_idx - bp_delay (raw-signal time)
    searchback: bool
    emitted_at: int      # sample index at which the decision was made
    r_refined: int = -1
    edge: bool = False


# ---------------------------------------------------------------- front-end

def _running_sum(a: np.ndarray, n: int) -> np.ndarray:
    c = np.cumsum(a, dtype=np.int64)
    out = c.copy()
    out[n:] -= c[:-n]
    return out


def _delay(a: np.ndarray, k: int) -> np.ndarray:
    out = np.zeros_like(a)
    out[k:] = a[:len(a) - k]
    return out


def frontend_offline(adc: np.ndarray, adc_zero: int, cfg: PTConfig) -> dict[str, np.ndarray]:
    """Vectorised integer front-end (int64). Bit-identical to FrontEnd.push."""
    x = np.asarray(adc, dtype=np.int64) - int(adc_zero)
    s1 = _running_sum(x, cfg.lp_len)
    lp = _running_sum(s1, cfg.lp_len)
    hp_sum = _running_sum(lp, cfg.hp_len)
    hp_scaled = cfg.hp_len * _delay(lp, cfg.hp_delay)
    hp = hp_scaled - hp_sum
    d = 2 * hp + _delay(hp, 1) - _delay(hp, 3) - 2 * _delay(hp, 4)
    q = d >> cfg.deriv_shift
    sq = q * q
    mwi = _running_sum(sq, cfg.mwi_len)
    return dict(x=x, s1=s1, lp=lp, hp_sum=hp_sum, hp_scaled=hp_scaled, hp=hp,
                d=d, q=q, sq=sq, mwi=mwi)


class _RunningSum:
    __slots__ = ("buf", "pos", "total")

    def __init__(self, n: int):
        self.buf = [0] * n
        self.pos = 0
        self.total = 0

    def push(self, v: int) -> int:
        self.total += v - self.buf[self.pos]
        self.buf[self.pos] = v
        self.pos = (self.pos + 1) % len(self.buf)
        return self.total


class FrontEnd:
    """Per-sample integer front-end: O(1) running sums for LP, HP and MWI."""

    def __init__(self, cfg: PTConfig, adc_zero: int):
        self.cfg = cfg
        self.adc_zero = int(adc_zero)
        self._s1 = _RunningSum(cfg.lp_len)
        self._lp = _RunningSum(cfg.lp_len)
        self._hp = _RunningSum(cfg.hp_len)     # buf holds lp history (len hp_len)
        self._mwi = _RunningSum(cfg.mwi_len)
        self._h = [0, 0, 0, 0]                 # hp[n-1..n-4]

    def push(self, adc: int) -> tuple[int, ...]:
        cfg = self.cfg
        x = int(adc) - self.adc_zero
        s1 = self._s1.push(x)
        lp = self._lp.push(s1)
        hp_sum = self._hp.push(lp)
        # after push, lp[n-k] sits at (pos - 1 - k) mod hp_len
        hp_scaled = cfg.hp_len * self._hp.buf[(self._hp.pos - 1 - cfg.hp_delay) % cfg.hp_len]
        hp = hp_scaled - hp_sum
        h1, h2, h3, h4 = self._h
        d = 2 * hp + h1 - h3 - 2 * h4
        self._h = [hp, h1, h2, h3]
        q = d >> cfg.deriv_shift
        sq = q * q
        mwi = self._mwi.push(sq)
        return x, s1, lp, hp_sum, hp_scaled, hp, d, q, sq, mwi


# ---------------------------------------------------------------- decision logic

class _Peak:
    __slots__ = ("p", "mwi", "bp_idx", "bp", "slope")

    def __init__(self, p, mwi, bp_idx, bp, slope):
        self.p, self.mwi, self.bp_idx, self.bp, self.slope = p, mwi, bp_idx, bp, slope


class _Ring8:
    __slots__ = ("buf", "pos", "total")

    def __init__(self, v: int):
        self.fill(v)

    def fill(self, v: int):
        self.buf = [v] * 8
        self.pos = 0
        self.total = 8 * v

    def push(self, v: int):
        self.total += v - self.buf[self.pos]
        self.buf[self.pos] = v
        self.pos = (self.pos + 1) & 7

    def copy_from(self, other: "_Ring8"):
        self.buf = list(other.buf)
        self.pos = other.pos
        self.total = other.total

    @property
    def mean(self) -> int:
        return self.total >> 3


def _missed_limit(avg: int) -> int:
    return avg + (avg >> 1) + (avg >> 3) + (avg >> 5)     # x 1.65625


def _rr_limits(avg: int) -> tuple[int, int]:
    low = avg - (avg >> 4) - (avg >> 6)                   # x 0.921875
    high = avg + (avg >> 3) + (avg >> 5)                  # x 1.15625
    return low, high


class DecisionEngine:
    """Adaptive-threshold decision stage; consumes (hp, d, mwi) one sample at a time.

    Shared by the streaming and the offline detector, so both make identical decisions.
    """

    def __init__(self, cfg: PTConfig):
        self.cfg = cfg
        self.n = -1
        self._mask = cfg.hist_len - 1
        if cfg.hist_len & self._mask:
            raise ValueError("hist_len must be a power of 2")
        if cfg.hist_len <= cfg.peak_confirm + cfg.pair_back:
            raise ValueError("hist_len too small for the pairing window")
        self._hp_abs = [0] * cfg.hist_len
        self._d_abs = [0] * cfg.hist_len
        self._m1 = 0                      # mwi[n-1]
        self._m2 = 0                      # mwi[n-2]
        self._cand_idx = -1
        self._cand_val = -1
        # learning
        self._learn_start = cfg.settle
        self._learn_end = cfg.settle + cfg.learn_len - 1
        self.learned = False
        self._lmax_i = self._lsum_i = self._lmax_f = self._lsum_f = 0
        self._queue: list[_Peak] = []
        # thresholds
        self.spki = self.npki = self.spkf = self.npkf = 0
        # beats / RR
        self._last_ref = 0                # timing reference for search-back
        self._last_beat = None            # bp_idx of last accepted beat
        self._last_slope = 0
        self._rr1 = _Ring8(cfg.fs)
        self._rr2 = _Ring8(cfg.fs)
        self._has_rr = False
        self._out_count = 0
        self.irregular = False
        self.rr_missed = _missed_limit(cfg.fs)
        self._sb: list[_Peak] = []
        self._sb_done = False

    # ---- per-sample entry point
    def process(self, hp: int, d: int, mwi: int) -> list[Detection]:
        cfg = self.cfg
        self.n += 1
        n = self.n
        k = n & self._mask
        self._hp_abs[k] = hp if hp >= 0 else -hp
        self._d_abs[k] = d if d >= 0 else -d
        out: list[Detection] = []

        m1 = self._m1
        if n >= 2 and m1 > self._m2 and m1 >= mwi and m1 > self._cand_val:
            self._cand_idx, self._cand_val = n - 1, m1
        self._m2, self._m1 = m1, mwi

        if self._cand_idx >= 0 and n - self._cand_idx >= cfg.peak_confirm:
            self._confirm(out)

        if self._learn_start <= n <= self._learn_end:
            if mwi > self._lmax_i:
                self._lmax_i = mwi
            self._lsum_i += mwi
            a = self._hp_abs[k]
            if a > self._lmax_f:
                self._lmax_f = a
            self._lsum_f += a
            if n == self._learn_end:
                self._finish_learning(out)

        if (self.learned and not self._sb_done
                and n - self._last_ref > self.rr_missed + cfg.peak_confirm + cfg.pair_back):
            self._searchback(out)
        return out

    def flush(self) -> list[Detection]:
        out: list[Detection] = []
        if self._cand_idx >= 0:
            self._confirm(out)
        if not self.learned and self.n >= self._learn_start:
            self._finish_learning(out)
        return out

    # ---- internals
    def _confirm(self, out):
        p, val = self._cand_idx, self._cand_val
        self._cand_idx, self._cand_val = -1, -1
        if p < self.cfg.settle:
            return                         # start-up transient of the FIR stages
        pk = self._make_peak(p, val)
        if self.learned:
            self._classify(pk, out)
        else:
            self._queue.append(pk)

    def _make_peak(self, p: int, val: int) -> _Peak:
        cfg, mask = self.cfg, self._mask
        lo = max(p - cfg.pair_back, 0)
        hi = p - cfg.deriv_delay
        bp_idx, bp = lo, -1
        for i in range(lo, hi + 1):
            v = self._hp_abs[i & mask]
            if v > bp:
                bp_idx, bp = i, v
        slope = 0
        for i in range(max(p - cfg.mwi_len + 1, 0), p + 1):
            v = self._d_abs[i & mask]
            if v > slope:
                slope = v
        return _Peak(p, val, bp_idx, bp, slope)

    def _finish_learning(self, out):
        cfg = self.cfg
        self.spki = self._lmax_i >> 2
        self.npki = (self._lsum_i // cfg.learn_len) >> 1
        self.spkf = self._lmax_f >> 2
        self.npkf = (self._lsum_f // cfg.learn_len) >> 1
        self.learned = True
        self._last_ref = self.n
        queue, self._queue = self._queue, []
        for pk in queue:
            self._classify(pk, out)

    def _thresholds(self) -> tuple[int, int]:
        thr_i = self.npki + ((self.spki - self.npki) >> 2)
        thr_f = self.npkf + ((self.spkf - self.npkf) >> 2)
        if self.irregular and self.cfg.irregular_halving:
            thr_i >>= 1
            thr_f >>= 1
        return thr_i, thr_f

    def _is_twave(self, pk: _Peak) -> bool:
        return (self._last_beat is not None
                and pk.bp_idx - self._last_beat < self.cfg.twave_window
                and pk.slope < (self._last_slope >> 1))

    def _classify(self, pk: _Peak, out):
        if self._last_beat is not None and pk.bp_idx - self._last_beat < self.cfg.refractory:
            return                         # refractory: ignored, no threshold update
        thr_i, thr_f = self._thresholds()
        is_qrs = pk.mwi > thr_i and pk.bp > thr_f and not self._is_twave(pk)
        if is_qrs:
            self.spki += (pk.mwi - self.spki) >> 3
            self.spkf += (pk.bp - self.spkf) >> 3
            self._accept(pk, False, out)
        else:
            self.npki += (pk.mwi - self.npki) >> 3
            self.npkf += (pk.bp - self.npkf) >> 3
            self._sb.append(pk)
            if len(self._sb) > self.cfg.sb_buffer:
                self._sb.pop(0)

    def _searchback(self, out):
        thr_i, thr_f = self._thresholds()
        thr_i2, thr_f2 = thr_i >> 1, thr_f >> 1
        best = None
        for pk in self._sb:
            if self._last_beat is not None:
                if pk.bp_idx - self._last_beat < self.cfg.refractory or self._is_twave(pk):
                    continue
            if pk.mwi > thr_i2 and pk.bp > thr_f2 and (best is None or pk.mwi > best.mwi):
                best = pk
        if best is None:
            self._sb_done = True
            return
        self.spki += (best.mwi - self.spki) >> 2
        self.spkf += (best.bp - self.spkf) >> 2
        self._accept(best, True, out)

    def _accept(self, pk: _Peak, searchback: bool, out):
        if self._last_beat is not None:
            self._update_rr(pk.bp_idx - self._last_beat)
        self._last_beat = pk.bp_idx
        self._last_ref = pk.bp_idx
        self._last_slope = pk.slope
        self._sb = [q for q in self._sb if q.bp_idx > pk.bp_idx]
        self._sb_done = False
        out.append(Detection(pk.bp_idx, pk.p, pk.bp_idx - self.cfg.bp_delay, searchback, self.n))

    def _update_rr(self, rr: int):
        if not self._has_rr:
            self._rr1.fill(rr)
            self._rr2.fill(rr)
            self._has_rr = True
            self.irregular = False
        else:
            low, high = _rr_limits(self._rr2.mean)
            self._rr1.push(rr)
            if low <= rr <= high:
                self._rr2.push(rr)
                self.irregular = False
                self._out_count = 0
            else:
                self.irregular = True
                self._out_count += 1
                if self._out_count >= 8:
                    self._rr2.copy_from(self._rr1)
                    self._out_count = 0
        self.rr_missed = _missed_limit(self._rr2.mean)


# ---------------------------------------------------------------- detectors

def _physical(adc_value, adc_zero: int, adc_gain: float):
    # Same expression as Record.signal (int arithmetic, then true division).
    return (adc_value - adc_zero) / adc_gain


class PanTompkinsStream:
    """Streaming detector: push one ADC sample, get finished detections (with refined R)."""

    def __init__(self, cfg: PTConfig, adc_zero: int, adc_gain: float,
                 sos: np.ndarray | None = None, record_stages: bool = False):
        from python.dsp.filters import SosStream
        self.cfg = cfg
        self.adc_zero, self.adc_gain = int(adc_zero), float(adc_gain)
        self.sos = design_bandpass(fs=cfg.fs) if sos is None else sos
        self.delay = group_delay_compensation(self.sos, cfg.fs)
        self.half = half_window_samples(cfg.fs)
        self._fe = FrontEnd(cfg, adc_zero)
        self._eng = DecisionEngine(cfg)
        self._bp = SosStream(self.sos)
        self._filt = [0.0] * cfg.filt_buffer
        self.n = -1
        self.stages: list[tuple] | None = [] if record_stages else None

    def push(self, adc: int) -> list[Detection]:
        self.n += 1
        st = self._fe.push(adc)
        if self.stages is not None:
            self.stages.append(st)
        self._filt[self.n % self.cfg.filt_buffer] = self._bp.push(
            _physical(int(adc), self.adc_zero, self.adc_gain))
        dets = self._eng.process(st[5], st[6], st[9])
        for det in dets:
            self._refine(det)
        return dets

    def flush(self) -> list[Detection]:
        dets = self._eng.flush()
        for det in dets:
            self._refine(det)
        return dets

    def _refine(self, det: Detection):
        c = det.r_coarse + self.delay
        lo, hi = max(c - self.half, 0), min(c + self.half, self.n)
        if lo <= self.n - self.cfg.filt_buffer:
            raise RuntimeError(f"refinement window at {lo} left the {self.cfg.filt_buffer}-sample buffer")
        window = np.array([self._filt[i % self.cfg.filt_buffer] for i in range(lo, hi + 1)])
        det.r_refined, det.edge = realign_one(det.r_coarse, window, self.delay, self.half,
                                              self.cfg.realign_mode, offset=lo)


@dataclass
class DetectorResult:
    detections: list[Detection]
    x_filt: np.ndarray
    delay: int
    stages: dict[str, np.ndarray] | None = None

    @property
    def r_coarse(self) -> np.ndarray:
        return np.array([d.r_coarse for d in self.detections], dtype=np.int64)

    @property
    def r_refined(self) -> np.ndarray:
        return np.array([d.r_refined for d in self.detections], dtype=np.int64)


def detect_offline(adc: np.ndarray, adc_zero: int, adc_gain: float, cfg: PTConfig,
                   sos: np.ndarray | None = None, return_stages: bool = False) -> DetectorResult:
    """Offline detector: vectorised front-end + the shared DecisionEngine + sosfilt refinement."""
    sos = design_bandpass(fs=cfg.fs) if sos is None else sos
    delay = group_delay_compensation(sos, cfg.fs)
    half = half_window_samples(cfg.fs)
    st = frontend_offline(adc, adc_zero, cfg)
    eng = DecisionEngine(cfg)
    dets: list[Detection] = []
    for hp, d, mwi in zip(st["hp"].tolist(), st["d"].tolist(), st["mwi"].tolist()):
        r = eng.process(hp, d, mwi)
        if r:
            dets.extend(r)
    dets.extend(eng.flush())
    x_filt = apply_offline(_physical(np.asarray(adc, dtype=np.int64), adc_zero, adc_gain), sos)
    for det in dets:
        det.r_refined, det.edge = realign_one(det.r_coarse, x_filt, delay, half, cfg.realign_mode)
    return DetectorResult(dets, x_filt, delay, st if return_stages else None)


def detector_from_config(cfg: dict) -> tuple[PTConfig, np.ndarray]:
    """Build (PTConfig, 0.5-40 Hz SOS) from an experiment config dict."""
    pt = PTConfig.from_dict(cfg["pan_tompkins"])
    f = cfg["filter"]
    sos = design_bandpass(f["low_hz"], f["high_hz"], f["order"], fs=pt.fs)
    return pt, sos


def stage_max_abs(stages: dict[str, np.ndarray]) -> dict[str, int]:
    return {k: int(np.abs(stages[k]).max()) for k in STAGE_NAMES}
