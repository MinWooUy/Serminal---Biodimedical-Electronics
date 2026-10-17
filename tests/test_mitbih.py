import numpy as np
import pytest

from python.data.constants import (
    AAMI_MAP, ALL_RECORDS, DS1, DS2, DS2_ENV_VAR, NON_BEAT_SYMBOLS, PACED, Q_LABEL,
    DS2LockedError, require_ds2_access,
)
from python.data.mitbih import (
    check_ds1_sanity, count_labels, load_record, parse_annotations,
)


# ---------- record lists ----------

def test_record_lists():
    assert len(DS1) == 22 and len(set(DS1)) == 22
    assert len(DS2) == 22 and len(set(DS2)) == 22
    assert not set(DS1) & set(DS2)
    assert not set(PACED) & (set(DS1) | set(DS2))
    assert len(ALL_RECORDS) == 44


# ---------- AAMI mapping / annotation parsing ----------

@pytest.mark.parametrize("sym,label", [
    ("N", 0), ("L", 0), ("R", 0), ("e", 0), ("j", 0),
    ("A", 1), ("a", 1), ("J", 1), ("S", 1),
    ("V", 2), ("E", 2),
    ("F", 3),
    ("/", Q_LABEL), ("f", Q_LABEL), ("Q", Q_LABEL),
])
def test_aami_map(sym, label):
    assert AAMI_MAP[sym] == label


def test_parse_keeps_q_and_drops_non_beats():
    samples = [10, 20, 30, 40, 50, 60, 70, 80]
    symbols = ["N", "+", "Q", "~", "V", "x", "/", "!"]
    p = parse_annotations(samples, symbols, sig_len=100)
    assert p.r_samples.tolist() == [10, 30, 50, 70]
    assert p.symbols.tolist() == ["N", "Q", "V", "/"]
    assert p.aami.tolist() == [0, Q_LABEL, 2, Q_LABEL]
    assert p.aami.dtype == np.int8
    assert p.vf_segments.shape == (0, 2)


def test_parse_vf_segments():
    samples = [5, 10, 15, 20, 25, 30, 35, 40]
    symbols = ["N", "[", "!", "!", "]", "N", "[", "!"]
    p = parse_annotations(samples, symbols, sig_len=100)
    assert p.vf_segments.tolist() == [[10, 25], [35, 99]]  # unclosed -> last sample
    assert p.r_samples.tolist() == [5, 30]


def test_parse_unknown_symbol_raises():
    with pytest.raises(ValueError, match="unknown annotation symbol"):
        parse_annotations([10], ["B"], sig_len=100)


def test_parse_nested_vf_raises():
    with pytest.raises(ValueError, match="nested"):
        parse_annotations([10, 20], ["[", "["], sig_len=100)


def test_beat_and_non_beat_sets_disjoint():
    assert not set(AAMI_MAP) & NON_BEAT_SYMBOLS
    assert "!" in NON_BEAT_SYMBOLS


# ---------- DS2 lock ----------

@pytest.mark.parametrize("env", [{}, {DS2_ENV_VAR: "0"}, {DS2_ENV_VAR: "true"},
                                 {DS2_ENV_VAR: " 1"}, {DS2_ENV_VAR: "1 "},
                                 {DS2_ENV_VAR: ""}])
def test_ds2_lock_rejects(env):
    # Uses an injected mapping; os.environ is never modified.
    with pytest.raises(DS2LockedError):
        require_ds2_access(env)


@pytest.mark.parametrize("rec_id", DS2)
def test_load_record_ds2_locked_before_file_access(rec_id, tmp_path, monkeypatch):
    monkeypatch.delenv(DS2_ENV_VAR, raising=False)
    # Empty dir: a FileNotFoundError would mean the lock is checked too late.
    with pytest.raises(DS2LockedError):
        load_record(rec_id, tmp_path)


@pytest.mark.parametrize("rec_id", PACED + (999,))
def test_load_record_rejects_paced_and_unknown(rec_id, tmp_path):
    with pytest.raises(ValueError):
        load_record(rec_id, tmp_path)


# ---------- sanity helpers ----------

def test_check_ds1_sanity():
    ok = {"N": 45800, "S": 950, "V": 3790, "F": 414, "Q": 0}
    check_ds1_sanity(ok)
    with pytest.raises(ValueError):
        check_ds1_sanity(ok | {"F": 300})


# ---------- real data ----------

@pytest.mark.needs_data
def test_mlii_channel_record_114(ds1_records):
    rec = ds1_records[114]  # leads are swapped in 114 (V5 first, MLII second)
    assert rec.fs == 360
    assert rec.signal.ndim == 1 and len(rec.signal) == 650000
    # MLII R waves are mostly positive in 114: median amplitude at beats > 0
    assert np.median(rec.signal[rec.r_samples]) > 0


@pytest.mark.needs_data
def test_record_arrays_consistent(ds1_records):
    for rec in ds1_records.values():
        n = len(rec.r_samples)
        assert len(rec.symbols) == n and len(rec.aami) == n
        assert np.all(np.diff(rec.r_samples) > 0)
        assert set(np.unique(rec.aami)) <= {0, 1, 2, 3, Q_LABEL}


@pytest.mark.needs_data
def test_adc_fields(ds1_records):
    for rec in ds1_records.values():
        assert rec.adc.dtype == np.int16 and rec.adc_zero == 1024 and rec.adc_gain == 200.0
        assert rec.adc.min() >= 0 and rec.adc.max() <= 2048       # 11-bit ADC
        assert np.array_equal(rec.signal, (rec.adc.astype(np.int64) - rec.adc_zero) / rec.adc_gain)


@pytest.mark.needs_data
def test_record_207_has_vf(ds1_records):
    vf = ds1_records[207].vf_segments
    assert len(vf) >= 1
    assert np.all(vf[:, 0] < vf[:, 1])
    assert np.all((vf >= 0) & (vf < len(ds1_records[207].signal)))


@pytest.mark.needs_data
def test_ds1_sanity_counts(ds1_records):
    counts = count_labels(ds1_records.values())
    print("DS1 counts:", counts)
    check_ds1_sanity(counts)
