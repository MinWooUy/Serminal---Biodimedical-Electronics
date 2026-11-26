import numpy as np
import pytest

from python.dsp.evaluate_detector import match_beats, outside_vf, tol_samples
from python.dsp.pan_tompkins import DecisionEngine, PTConfig, frontend_offline
from python.viz.common import (
    FP, NOT_CLASSIFIED, VF, ThresholdTraceEngine, rr_features_seconds, softmax_from_int_logits, trace_detector,
)
from tests.test_pan_tompkins import ADC_ZERO, synthetic_ecg

CFG = PTConfig()


def _base_detections(adc, adc_zero, cfg):
    st = frontend_offline(adc, adc_zero, cfg)
    eng = DecisionEngine(cfg)
    dets = []
    for hp, d, mwi in zip(st["hp"].tolist(), st["d"].tolist(), st["mwi"].tolist()):
        dets += eng.process(hp, d, mwi)
    return dets + eng.flush()


@pytest.mark.parametrize("seed,small_beat", [(1, None), (0, 20)])
def test_trace_engine_detections_identical_synthetic(seed, small_beat):
    adc, _ = synthetic_ecg(seed=seed, small_beat=small_beat)
    tr = trace_detector(adc, ADC_ZERO, CFG)
    assert tr["detections"] == _base_detections(adc, ADC_ZERO, CFG)
    assert len(tr["thr_i1"]) == len(tr["mwi"])
    assert np.array_equal(tr["thr_i2"], tr["thr_i1"] >> 1)
    # every accepted peak (non search-back) is above THR1 at its classification time; noise peaks recorded
    acc = [p for p in tr["peaks"] if p[3]]
    assert len(acc) == sum(not d.searchback for d in tr["detections"])
    if small_beat is not None:
        assert tr["searchback"].sum() == 1


def test_trace_engine_thresholds_match_engine_state():
    adc, _ = synthetic_ecg(seed=2)
    st = frontend_offline(adc, ADC_ZERO, CFG)
    eng = ThresholdTraceEngine(CFG)
    for hp, d, mwi in zip(st["hp"].tolist(), st["d"].tolist(), st["mwi"].tolist()):
        eng.process(hp, d, mwi)
    assert eng.thr_i1[-1] == eng._thresholds()[0] and eng.thr_f1[-1] == eng._thresholds()[1]
    assert eng.thr_i1[0] == 0 and max(eng.thr_i1) > 0


def test_softmax_from_int_logits():
    class _Q:
        fc2 = {"k": 6, "a_in": 4}
    logits = np.array([[1000, -200, 30, 5], [-7, -7, 900, 0]], dtype=np.int64)
    p = softmax_from_int_logits(logits, _Q())
    assert np.allclose(p.sum(1), 1.0)
    assert np.array_equal(p.argmax(1), logits.argmax(1))


def test_rr_features_seconds():
    parts = np.array([[360, 720, 8 * 360, 8], [180, 360, 3 * 540, 3]])
    f = rr_features_seconds(parts, 360)
    assert np.allclose(f[0], [1.0, 2.0, 1.0, 1.0])
    assert np.allclose(f[1], [0.5, 1.0, 1.5, 1 / 3])


# ---------- real data (frozen model) ----------

@pytest.mark.needs_data
def test_trace_engine_identical_ds1(ds1_records):
    rec = ds1_records[208]
    tr = trace_detector(rec.adc, rec.adc_zero, CFG)
    assert tr["detections"] == _base_detections(rec.adc, rec.adc_zero, CFG)


@pytest.mark.needs_data
@pytest.mark.parametrize("rec_id", [208])
def test_record_view_consistent_with_detector_evaluation(ds1_records, rec_id):
    from python.viz.common import load_frozen, record_view
    rec = ds1_records[rec_id]
    fz = load_frozen()
    v = record_view(rec, fz, "detector")
    ka = outside_vf(rec.r_samples, rec.vf_segments)
    kd = outside_vf(v["r"], rec.vf_segments)
    m = match_beats(rec.r_samples[ka], v["r"][kd], tol_samples(150, fz.fs))
    assert (v["label"] == FP).sum() == len(m.fp)
    assert v["ann_missed"].sum() == len(m.fn)
    assert (v["label"] == VF).sum() == (~kd).sum()
    # every beat with a complete window is classified, and only those
    cls = v["pred"] != NOT_CLASSIFIED
    assert np.array_equal(np.flatnonzero(cls), v["beat_index"])
    assert np.allclose(v["probs"][cls].sum(1), 1.0)

    va = record_view(rec, fz, "annotation")
    assert len(va["r"]) == len(rec.r_samples) and not va["ann_missed"].any()
    assert np.array_equal(np.sort(va["label"]), np.sort(rec.aami))


def test_oof_fold_assignment():
    from python.data.constants import DS1, DS2
    from python.data.splits import load_folds
    from python.viz.common import oof_fold
    folds = load_folds()
    for rid in DS1:
        assert rid in folds[oof_fold(rid)]
    with pytest.raises(ValueError):
        oof_fold(DS2[0])


@pytest.mark.needs_data
def test_oof_view_matches_integer_gate(ds1_records):
    """view_record --model oof reproduces the per-beat (iii) logits of pipeline_int_eval exactly."""
    from python.viz.common import DEFAULT_CV_RUN, load_frozen, oof_quant_model, record_view
    f = DEFAULT_CV_RUN / "pipeline_int_preds.npz"
    if not f.exists():
        pytest.skip("pipeline_int_preds.npz not generated")
    d = np.load(f)
    rid = 208
    qm, _ = oof_quant_model(rid)
    v = record_view(ds1_records[rid], load_frozen(), "detector", qm=qm)
    sel = d["rec_id"] == rid
    bi = v["beat_index"]
    r = v["r"][bi]
    j = np.array([int(np.argmin(np.abs(r - x))) for x in d["r_pos"][sel]])
    assert np.array_equal(v["logits"][bi[j]], d["logits_iii_s42"][sel])
