import numpy as np
import pytest

from python.data.beats import (
    KIND_LABELED, BeatConfig, RRStream, build_record, cache_config, extract_windows, load_beats,
    normalize_beats, rr_clip_samples, rr_features, save_record, assert_ds1_only,
)
from python.data.constants import AAMI_MAP, DS2LockedError, Q_LABEL
from python.data.mitbih import Record, load_record
from python.dsp.filters import design_bandpass
from python.dsp.pan_tompkins import PTConfig
from python.experiment import config_hash
from tests.test_pan_tompkins import ADC_GAIN, ADC_ZERO, synthetic_ecg

FS = 360
PT = PTConfig()
SOS = design_bandpass()


def _r_at_180(X):
    a = np.abs(X)
    return a[:, 180] >= a[:, 162:199].max(axis=1)


# ---------- RR features ----------

def test_rr_known_sequence():
    r = np.array([0, 360, 720, 900, 1260])
    f = rr_features(r, FS)
    np.testing.assert_allclose(f[0], [1.0, 1.0, 1.0, 1.0])            # beat 1
    np.testing.assert_allclose(f[1], [1.0, 0.5, 1.0, 1.0])            # beat 2
    np.testing.assert_allclose(f[2], [0.5, 1.0, 900 / 3 / 360, 180 / 300])   # beat 3: avg of 3 past RRs
    assert f.shape == (3, 4)                                          # first and last beat dropped


def test_rr_avg_uses_at_most_8_past():
    r = np.cumsum(np.r_[0, [100] * 8, [300] * 3])      # beats 0..11
    f = rr_features(r, FS)
    i = 10                                              # past RRs ending at beat 10: rr[2..9]
    rr = np.diff(r)
    np.testing.assert_allclose(f[i - 1, 2], rr[2:10].mean() / FS)


@pytest.mark.parametrize("clip", [None, (72, 1080)])
def test_rr_stream_equals_offline_bitwise(clip):
    rng = np.random.default_rng(0)
    r = np.cumsum(rng.integers(20, 1500, size=300))     # includes RRs outside [72, 1080]
    off = rr_features(r, FS, rr_clip=clip)
    st = RRStream(FS, rr_clip=clip)
    got = [o for o in (st.push(int(v)) for v in r) if o is not None]
    assert [i for i, _ in got] == list(range(1, len(r) - 1))
    assert np.array_equal(np.array([f for _, f in got]), off)


@pytest.mark.parametrize("clip", [None, (72, 1080)])
def test_rr6_stream_equals_offline_bitwise(clip):
    rng = np.random.default_rng(5)
    r = np.cumsum(rng.integers(20, 1500, size=400))
    off = rr_features(r, FS, rr_clip=clip, avg_long_n=32)
    assert off.shape == (len(r) - 2, 6)
    np.testing.assert_array_equal(off[:, :4], rr_features(r, FS, rr_clip=clip))   # first 4 unchanged
    st = RRStream(FS, rr_clip=clip, avg_long_n=32)
    got = [o for o in (st.push(int(v)) for v in r) if o is not None]
    assert np.array_equal(np.array([f for _, f in got]), off)


def test_rr6_known_values():
    rr = [100] * 40 + [50, 200]                       # beats 0..42
    r = np.cumsum(np.r_[0, rr])
    f = rr_features(r, FS, avg_long_n=32)
    i = 41                                            # rr_prev = 50, rr_next = 200
    past32 = np.array(rr[i - 32:i])
    np.testing.assert_allclose(f[i - 1, 4], 50 / 200)
    np.testing.assert_allclose(f[i - 1, 5], 50 / past32.mean())
    np.testing.assert_allclose(f[0, 5], 1.0)          # beat 1: only one past RR (at least 1)


def test_rr_clip_applied_before_features():
    lo, hi = rr_clip_samples((0.2, 3.0), FS)
    assert (lo, hi) == (72, 1080)
    r = np.array([0, 360, 360 + 5000, 360 + 5000 + 30, 360 + 5000 + 30 + 360])
    f = rr_features(r, FS, rr_clip=(lo, hi))
    np.testing.assert_allclose(f[0, :2], [1.0, 3.0])                 # 5000 -> 1080 samples
    np.testing.assert_allclose(f[1], [3.0, 0.2, (360 + 1080) / 2 / FS, 1080 / 720])
    np.testing.assert_allclose(f[2, 0], 0.2)                         # 30 -> 72 samples
    assert f[:, :3].min() >= 0.2 and f[:, :3].max() <= 3.0


def test_rr_no_future_beyond_next():
    rng = np.random.default_rng(1)
    r = np.cumsum(rng.integers(100, 500, size=50))
    base = rr_features(r, FS)
    i = 20
    r2 = r.copy()
    r2[i + 2:] += 37                                   # change beats from i+2 on
    f2 = rr_features(r2, FS)
    assert np.array_equal(base[:i], f2[:i])            # rows of beats 1..i unchanged


# ---------- windows / normalisation ----------

def test_window_position_and_length():
    x = np.arange(2000, dtype=float)
    w, keep = extract_windows(x, np.array([100, 500, 1900]), delay=2, half=180)
    assert keep.tolist() == [False, True, False]       # incomplete windows dropped
    assert w.shape == (1, 360)
    assert w[0, 180] == 502 and w[0, 0] == 322 and w[0, -1] == 681


def test_window_ignores_samples_after_window():
    rng = np.random.default_rng(2)
    x = rng.normal(size=3000)
    w1, _ = extract_windows(x, np.array([1000]), 2, 180)
    x[1002 + 180:] = 99.0                              # everything after the window
    w2, _ = extract_windows(x, np.array([1000]), 2, 180)
    assert np.array_equal(w1, w2)


def test_zscore_and_clip():
    rng = np.random.default_rng(3)
    w = rng.normal(5, 2, size=(4, 360))
    w[0, 10] = 1e3
    z = normalize_beats(w, 1e-6, 5.0)
    assert np.all(np.abs(z) <= 5.0) and z[0, 10] == 5.0
    np.testing.assert_allclose(z[1:].mean(axis=1), 0, atol=1e-9)
    np.testing.assert_allclose(z[1:].std(axis=1), 1, atol=1e-5)
    assert np.all(np.isfinite(normalize_beats(np.ones((1, 360)), 1e-6, 5.0)))   # flat beat


# ---------- synthetic record, both sources ----------

def _synthetic_record():
    adc, r_pos = synthetic_ecg(seed=4)
    symbols = np.array(["N"] * len(r_pos), dtype="<U1")
    symbols[10], symbols[15] = "Q", "V"
    return Record(rec_id=101, fs=FS, signal=(adc - ADC_ZERO) / ADC_GAIN, r_samples=r_pos,
                  symbols=symbols, aami=np.array([AAMI_MAP[s] for s in symbols], dtype=np.int8),
                  vf_segments=np.zeros((0, 2), dtype=np.int64), adc=adc.astype(np.int16),
                  adc_zero=ADC_ZERO, adc_gain=ADC_GAIN)


@pytest.mark.parametrize("source", ["detector", "annotation"])
def test_build_record_both_sources(source):
    rec = _synthetic_record()
    arr, stats = build_record(rec, PT, SOS, BeatConfig(rr_source=source))
    n = len(rec.r_samples)
    assert arr["X"].shape == (n - 2, 360) and arr["rr"].shape == (n - 2, 4)
    assert np.all(arr["kind"] == KIND_LABELED) and stats["fp"] == 0
    assert sum(stats["fn"].values()) == 0
    # Q kept in the cache (and in the RR sequence) but excluded from train/val
    assert np.sum(arr["y"] == Q_LABEL) == 1
    # beat after the Q beat: its rr_prev is the RR from the Q beat
    j = 10                                              # beat 11 sits at row 10
    assert abs(arr["rr"][j, 0] - (rec.r_samples[11] - rec.r_samples[10]) / FS) < 0.06
    # R at sample 180: |X| at 180 equals the max within +/-18 (ties allowed: z-score clipping
    # flattens tall R peaks)
    assert np.mean(_r_at_180(arr["X"])) > 0.95


def test_sources_agree_on_synthetic():
    rec = _synthetic_record()
    a, _ = build_record(rec, PT, SOS, BeatConfig(rr_source="annotation"))
    d, _ = build_record(rec, PT, SOS, BeatConfig(rr_source="detector"))
    assert np.array_equal(a["y"], d["y"])
    assert np.max(np.abs(a["r_pos"] - d["r_pos"])) <= 3


def test_cache_roundtrip_and_filter(tmp_path):
    rec = _synthetic_record()
    arr, stats = build_record(rec, PT, SOS, BeatConfig())
    save_record(tmp_path, rec.rec_id, arr, stats)
    all_ = load_beats(tmp_path, [rec.rec_id], trainable_only=False)
    tr = load_beats(tmp_path, [rec.rec_id])
    for k in arr:
        assert np.array_equal(all_[k], arr[k])
    assert len(tr["y"]) == len(arr["y"]) - 1 and np.all(tr["y"] >= 0)


def test_cache_hash_tracks_config():
    cfg = {"beats": {"rr_source": "detector", "half_window": 180}}
    det = {"pan_tompkins": {"irregular_halving": True}}
    h = config_hash(cache_config(cfg, det, "detector"))
    assert h != config_hash(cache_config(cfg, det, "annotation"))
    assert h != config_hash(cache_config(cfg, {"pan_tompkins": {"irregular_halving": False}}, "detector"))
    assert h == config_hash(cache_config(cfg, det, "detector"))


def test_ds2_locked(monkeypatch):
    monkeypatch.delenv("ECG_ALLOW_DS2", raising=False)
    with pytest.raises(ValueError):
        assert_ds1_only([101, 100])
    with pytest.raises(DS2LockedError):
        load_record(100)


# ---------- real data ----------

@pytest.mark.needs_data
@pytest.mark.parametrize("source", ["detector", "annotation"])
def test_real_record_r_at_180(ds1_records, source):
    arr, _ = build_record(ds1_records[208], PT, SOS, BeatConfig(rr_source=source))
    hit = _r_at_180(arr["X"])
    assert hit.mean() > 0.95, hit.mean()
