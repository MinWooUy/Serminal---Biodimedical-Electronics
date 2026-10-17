import numpy as np
import pytest

from python.data.constants import DS1
from python.dsp.evaluate_detector import match_beats
from python.dsp.pan_tompkins import (
    STAGE_NAMES, FrontEnd, PanTompkinsStream, PTConfig, detect_offline, frontend_offline,
    stage_max_abs,
)

FS = 360
ADC_ZERO, ADC_GAIN = 1024, 200.0
CFG = PTConfig()


def _gauss(n, center, sigma, amp):
    t = np.arange(n)
    return amp * np.exp(-0.5 * ((t - center) / sigma) ** 2)


def synthetic_ecg(seed=0, small_beat=None):
    """ADC counts with P, QRS, T waves at known R positions (variable RR)."""
    rng = np.random.default_rng(seed)
    rr = rng.integers(250, 330, size=40)           # 0.69-0.92 s
    r_pos = 400 + np.cumsum(rr)
    n = int(r_pos[-1] + 400)
    x = np.zeros(n)
    for i, r in enumerate(r_pos):
        # small beat: MWI ~0.17*SPKI, between THR2 (~0.13) and THR1 (~0.26) -> search-back only
        amp = 0.5 if i == small_beat else 1.0
        x += _gauss(n, r, 0.008 * FS, amp)                     # R
        x += _gauss(n, r - 0.012 * FS, 0.006 * FS, -0.15)     # Q
        x += _gauss(n, r + 0.014 * FS, 0.006 * FS, -0.2)      # S
        x += _gauss(n, r - 0.16 * FS, 0.02 * FS, 0.12)        # P
        x += _gauss(n, r + 0.28 * FS, 0.045 * FS, 0.3)        # T
    x += 0.2 * np.sin(2 * np.pi * 0.3 * np.arange(n) / FS)   # baseline wander
    x += rng.normal(0, 0.01, n)
    adc = np.round(x * ADC_GAIN).astype(np.int64) + ADC_ZERO
    return adc, r_pos


def _stream(adc, cfg=CFG, record_stages=False):
    st = PanTompkinsStream(cfg, ADC_ZERO, ADC_GAIN, record_stages=record_stages)
    dets = []
    for v in adc:
        dets.extend(st.push(int(v)))
    dets.extend(st.flush())
    return st, dets


# ---------- config / conversion ----------

def test_conversion_constants():
    assert (CFG.lp_len, CFG.hp_len, CFG.mwi_len) == (11, 57, 54)
    assert (CFG.refractory, CFG.twave_window, CFG.learn_len) == (72, 130, 720)
    assert (CFG.lp_delay, CFG.hp_delay, CFG.bp_delay) == (10, 28, 38)


def test_threshold_update_identity():
    # (peak + 7*S) >> 3 == S + ((peak - S) >> 3), also for (peak + 3*S) >> 2
    rng = np.random.default_rng(0)
    for p, s in rng.integers(0, 2**29, size=(5000, 2)).tolist():
        assert (p + 7 * s) >> 3 == s + ((p - s) >> 3)
        assert (p + 3 * s) >> 2 == s + ((p - s) >> 2)


# ---------- synthetic signal ----------

def test_synthetic_perfect_detection():
    adc, r_pos = synthetic_ecg()
    res = detect_offline(adc, ADC_ZERO, ADC_GAIN, CFG)
    m = match_beats(r_pos, res.r_refined, tol=27)
    assert m.tp == len(r_pos) and len(m.fn) == 0 and len(m.fp) == 0


def test_mwi_and_bandpass_peaks_paired():
    adc, r_pos = synthetic_ecg()
    res = detect_offline(adc, ADC_ZERO, ADC_GAIN, CFG)
    for d in res.detections:
        # band-pass peak lies in the MWI integration window and on the same beat
        assert d.mwi_idx - CFG.pair_back <= d.bp_idx <= d.mwi_idx - CFG.deriv_delay
        beat_bp = np.argmin(np.abs(r_pos - (d.bp_idx - CFG.bp_delay)))
        beat_mwi = np.argmin(np.abs(r_pos - (d.mwi_idx - 66)))
        assert beat_bp == beat_mwi
        assert abs(d.bp_idx - CFG.bp_delay - r_pos[beat_bp]) <= 2


def test_searchback_recovers_small_beat():
    adc, r_pos = synthetic_ecg(small_beat=20)
    res = detect_offline(adc, ADC_ZERO, ADC_GAIN, CFG)
    m = match_beats(r_pos, res.r_refined, tol=27)
    assert m.tp == len(r_pos) and len(m.fp) == 0
    j = m.pairs[m.pairs[:, 0] == 20, 1][0]
    assert res.detections[j].searchback


def test_refractory_respected():
    adc, _ = synthetic_ecg(seed=3)
    res = detect_offline(adc, ADC_ZERO, ADC_GAIN, CFG)
    assert np.all(np.diff([d.bp_idx for d in res.detections]) >= CFG.refractory)


def test_stream_equals_offline_synthetic():
    adc, _ = synthetic_ecg(seed=1)
    st, dets = _stream(adc, record_stages=True)
    res = detect_offline(adc, ADC_ZERO, ADC_GAIN, CFG, return_stages=True)
    stages = np.array(st.stages, dtype=np.int64)
    for i, name in enumerate(STAGE_NAMES):
        assert np.array_equal(stages[:, i], res.stages[name]), name
    assert dets == res.detections


def test_frontend_running_sums_match():
    adc = np.random.default_rng(2).integers(0, 2048, size=3000)
    fe = FrontEnd(CFG, ADC_ZERO)
    stream = np.array([fe.push(int(v)) for v in adc], dtype=np.int64)
    off = frontend_offline(adc, ADC_ZERO, CFG)
    for i, name in enumerate(STAGE_NAMES):
        assert np.array_equal(stream[:, i], off[name]), name


# ---------- real data ----------

@pytest.mark.needs_data
@pytest.mark.parametrize("rec_id", [108, 207, 208])
def test_stream_equals_offline_bitwise_ds1(ds1_records, rec_id):
    rec = ds1_records[rec_id]
    st = PanTompkinsStream(CFG, rec.adc_zero, rec.adc_gain, record_stages=True)
    dets = []
    for v in rec.adc.tolist():
        dets.extend(st.push(v))
    dets.extend(st.flush())
    res = detect_offline(rec.adc, rec.adc_zero, rec.adc_gain, CFG, return_stages=True)
    stages = np.array(st.stages, dtype=np.int64)
    for i, name in enumerate(STAGE_NAMES):
        assert np.array_equal(stages[:, i], res.stages[name]), name
    assert dets == res.detections          # incl. r_coarse, r_refined, edge, searchback


@pytest.mark.needs_data
def test_int32_budget_ds1(ds1_records):
    maxima = {name: 0 for name in STAGE_NAMES}
    for rec in ds1_records.values():
        for k, v in stage_max_abs(frontend_offline(rec.adc, rec.adc_zero, CFG)).items():
            maxima[k] = max(maxima[k], v)
    print("\nstage        max|value|     log2   (limit 2^31)")
    for k, v in maxima.items():
        print(f"{k:10s} {v:14d}  {np.log2(max(v, 1)):6.2f}")
    assert all(v < 2**31 for v in maxima.values())
