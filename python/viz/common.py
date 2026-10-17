"""Shared, READ-ONLY helpers for the visualisation tools (frozen models; nothing here trains or tunes).

- load_frozen(): the frozen Lite int8 parameters (lite_int8.npz, SHA-256 checked against
  export/final_models_manifest.json) and the frozen pipeline configuration.
- ThresholdTraceEngine: DecisionEngine subclass that only RECORDS the adaptive thresholds and noise peaks;
  its detections are identical to the base engine (tested).
- oof_quant_model(): for a DS1 record, the int8 model of the CV fold that has the record in its OUTER fold
  (seed 42 by default), rebuilt exactly as scripts/pipeline_int_eval.py does (case iii) and cached.
- record_view(): the fully integer pipeline on one record (detector or annotation segmentation),
  with per-detection labels / predictions / probabilities / RR features and the FN annotations.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from python.data.beats import CACHE_ROOT
from python.data.constants import DS1
from python.data.mitbih import REPO_ROOT, Record
from python.dsp.evaluate_detector import match_beats, outside_vf, tol_samples
from python.dsp.filters import group_delay_compensation
from python.dsp.pan_tompkins import DecisionEngine, PTConfig, detector_from_config, frontend_offline
from python.experiment import load_config
from python.quant.int8 import QuantModel, int_forward_q, load_quant_model
from python.quant.pipeline_int import FILTER_GUARD_BITS, filter_int, run_integer_beats

DEFAULT_CV_RUN = REPO_ROOT / "results" / "2026-10-15_a50fb3e1_A5_svdb_n25_lite_e3"     # chosen Lite CV run
CLASSES = ("N", "S", "V", "F")
FP, VF, Q, NOT_CLASSIFIED = -2, -3, -1, -9


@dataclass
class Frozen:
    qm: QuantModel
    lite_dir: Path
    pt: PTConfig
    sos: np.ndarray
    delay: int
    half: int
    rr_lo: int
    rr_hi: int
    fs: int


def load_frozen(verify: bool = True) -> Frozen:
    man = json.loads((REPO_ROOT / "export" / "final_models_manifest.json").read_text(encoding="utf-8"))
    lite_dir = REPO_ROOT / "results" / man["lite"]["dir"]
    npz = lite_dir / "lite_int8.npz"
    if verify and hashlib.sha256(npz.read_bytes()).hexdigest() != man["lite"]["files"]["lite_int8.npz"]:
        raise RuntimeError(f"{npz} does not match export/final_models_manifest.json (frozen model changed?)")
    bcfg = load_config(REPO_ROOT / load_config(lite_dir / "config.yaml")["beats_config"])
    pt, sos = detector_from_config(load_config(REPO_ROOT / bcfg["detector_config"]))
    lo, hi = (int(round(v * pt.fs)) for v in bcfg["beats"]["rr_clip_s"])
    return Frozen(load_quant_model(npz), lite_dir, pt, sos, group_delay_compensation(sos, pt.fs),
                  int(bcfg["beats"]["half_window"]), lo, hi, pt.fs)


def oof_fold(rid: int, run: Path = DEFAULT_CV_RUN) -> int:
    from python.data.splits import load_folds
    cfg = load_config(run / "config.yaml")
    folds = load_folds()
    ks = [k for k in cfg["folds"] if rid in folds[k]]
    if rid not in DS1 or len(ks) != 1:
        raise ValueError(f"record {rid}: out-of-fold models exist only for DS1 records")
    return ks[0]


def oof_quant_model(rid: int, seed: int = 42, run: Path = DEFAULT_CV_RUN, n_calib: int = 4000) -> tuple[QuantModel, int]:
    """int8 model of the fold whose OUTER part contains record rid (same checkpoint, BN folding and
    calibration sample as pipeline_int_eval.py: rng(seed * 10 + k), n_calib inner-train beats incl. SVDB).
    Cached in data/cache/viz_oof/ (keyed by the checkpoint SHA-256). Returns (model, fold)."""
    k = oof_fold(rid, run)
    ckpt = run / "ckpt" / f"s{seed}_f{k}.pt"
    h = hashlib.sha256(ckpt.read_bytes()).hexdigest()[:16]
    cache = CACHE_ROOT / "viz_oof" / f"{run.name}_s{seed}_f{k}_c{n_calib}_{h}.npz"
    if cache.exists():
        return load_quant_model(cache), k
    import torch
    from python.models.cnn1d import ECGCNN, ModelConfig
    from python.models.cv_data import attach_extra_train, load_cv_data, with_extra_train
    from python.models.training import TrainConfig, fold_indices
    from python.quant.int8 import fold_model, quantize_model, save_quant_model
    cfg = load_config(run / "config.yaml")
    cv = load_cv_data(cfg["beats_config"])
    d, n_main, extra_idx, _ = attach_extra_train(cfg, cv.data)
    tcfg = TrainConfig.from_dict(cfg["train"])
    ck = torch.load(ckpt, weights_only=False)
    m = ECGCNN(ModelConfig.from_dict(ck["model"]))
    m.load_state_dict(ck["state_dict"])
    idx = fold_indices(d["rec_id"][:n_main], d["r_pos"][:n_main], cv.rec_len, cv.folds, k, cv.delay, cv.half, tcfg,
                       cv.fs)
    idx = with_extra_train(idx, extra_idx, n_main)
    rng = np.random.default_rng(seed * 10 + k)
    cal = np.sort(rng.choice(idx["train"], size=min(n_calib, len(idx["train"])), replace=False))
    qm = quantize_model(fold_model(m), d["X"][cal], d["rr"][cal], "max")
    cache.parent.mkdir(parents=True, exist_ok=True)
    save_quant_model(qm, cache)
    return load_quant_model(cache), k


class ThresholdTraceEngine(DecisionEngine):
    """Records THR1 / THR2 of both channels after every sample and the classified peaks. Read-only:
    no decision logic is overridden, so the detections equal DecisionEngine's."""

    def __init__(self, cfg: PTConfig):
        super().__init__(cfg)
        self.thr_i1, self.thr_f1 = [], []
        self.peaks = []                     # (mwi_idx, bp_idx, mwi, accepted_as_qrs)

    def process(self, hp, d, mwi):
        out = super().process(hp, d, mwi)
        if self.learned:
            ti, tf = self._thresholds()
        else:
            ti = tf = 0
        self.thr_i1.append(ti)
        self.thr_f1.append(tf)
        return out

    def _classify(self, pk, out):
        n_before = len(out)
        super()._classify(pk, out)
        self.peaks.append((pk.p, pk.bp_idx, pk.mwi, len(out) > n_before))


def trace_detector(adc: np.ndarray, adc_zero: int, pt: PTConfig) -> dict:
    """Integer Pan-Tompkins with threshold tracing. Returns detections + MWI + thresholds per sample."""
    st = frontend_offline(adc, adc_zero, pt)
    eng = ThresholdTraceEngine(pt)
    dets = []
    for hp, d, mwi in zip(st["hp"].tolist(), st["d"].tolist(), st["mwi"].tolist()):
        dets += eng.process(hp, d, mwi)
    dets += eng.flush()
    thr_i1 = np.asarray(eng.thr_i1 + [eng.thr_i1[-1]] * (len(st["mwi"]) - len(eng.thr_i1)), dtype=np.int64)
    return {"detections": dets, "mwi": st["mwi"], "thr_i1": thr_i1, "thr_i2": thr_i1 >> 1,
            "peaks": eng.peaks, "r_coarse": np.array([d.r_coarse for d in dets], dtype=np.int64),
            "searchback": np.array([d.searchback for d in dets], dtype=bool)}


def softmax_from_int_logits(logits: np.ndarray, qm: QuantModel) -> np.ndarray:
    """Display only: softmax of the dequantised int8-path logits (the prediction is argmax of the int logits)."""
    z = np.asarray(logits, dtype=np.float64) * 2.0 ** -(qm.fc2["k"] + qm.fc2["a_in"])
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def quantise_rr(parts: np.ndarray, rr_exp: np.ndarray, fs: int) -> np.ndarray:
    from scripts.pipeline_int_eval import quantise_rr_parts
    return quantise_rr_parts(parts, rr_exp, fs)


def rr_features_seconds(parts: np.ndarray, fs: int) -> np.ndarray:
    """(rr_prev, rr_next, rr_avg8, ratio) in seconds from the integer RR parts (display)."""
    p = np.asarray(parts, dtype=np.float64)
    return np.stack([p[:, 0] / fs, p[:, 1] / fs, p[:, 2] / p[:, 3] / fs, p[:, 0] * p[:, 3] / p[:, 2]], axis=1)


def record_view(rec: Record, fz: Frozen, seg: str = "detector", qm: QuantModel | None = None) -> dict:
    """Fully integer pipeline on one record (frozen Lite, or the int8 model qm, e.g. an OOF fold model).
    seg: 'detector' (deployment) or 'annotation'.

    Returns per-sample filtered signal (mV) and, per R position (detector detections or annotations):
    position, label (class 0..3, Q -1, FP -2, inside VF -3), prediction (or -9), probabilities,
    RR features; plus annotation positions with a 'missed' flag (detector segmentation only),
    MWI / thresholds (detector) and the VF segments.
    """
    y_int = filter_int(rec.adc, rec.adc_zero, fz.sos)
    sig = y_int / 2.0**FILTER_GUARD_BITS / rec.adc_gain
    out = {"fs": fz.fs, "signal_mv": sig, "vf": rec.vf_segments, "ann_pos": rec.r_samples, "ann_label": rec.aami,
           "seg": seg}
    if seg == "detector":
        tr = trace_detector(rec.adc, rec.adc_zero, fz.pt)
        coarse = tr["r_coarse"]
        out |= {"mwi": tr["mwi"], "thr_i1": tr["thr_i1"], "thr_i2": tr["thr_i2"], "peaks": tr["peaks"]}
    elif seg == "annotation":
        coarse = rec.r_samples
    else:
        raise ValueError(f"unknown segmentation {seg!r}")
    ib = run_integer_beats(y_int, coarse, fz.delay, fz.half, fz.rr_lo, fz.rr_hi)
    r = ib["r_int"]
    n = len(r)
    pred = np.full(n, NOT_CLASSIFIED, dtype=np.int64)
    probs = np.full((n, 4), np.nan)
    logit = np.zeros((n, 4), dtype=np.int64)
    qm = fz.qm if qm is None else qm
    rr = np.full((n, 4), np.nan)
    bi = ib["beat_index"]
    if len(bi):
        logits = int_forward_q(qm, ib["z_q"], quantise_rr(ib["rr_parts"], qm.rr_exp, fz.fs))
        logit[bi] = logits
        pred[bi] = logits.argmax(1)
        probs[bi] = softmax_from_int_logits(logits, qm)
        rr[bi] = rr_features_seconds(ib["rr_parts"], fz.fs)
    if seg == "detector":
        ka, kd = outside_vf(rec.r_samples, rec.vf_segments), outside_vf(r, rec.vf_segments)
        ai, di = np.flatnonzero(ka), np.flatnonzero(kd)
        m = match_beats(rec.r_samples[ai], r[di], tol_samples(150, fz.fs))
        label = np.full(n, FP, dtype=np.int64)
        label[~kd] = VF
        label[di[m.pairs[:, 1]]] = rec.aami[ai[m.pairs[:, 0]]]
        missed = np.zeros(len(rec.r_samples), dtype=bool)
        missed[ai[m.fn]] = True
        sb = tr["searchback"][ib["order"]]
    else:
        label = rec.aami[ib["order"]].astype(np.int64)
        missed = np.zeros(len(rec.r_samples), dtype=bool)
        sb = np.zeros(n, dtype=bool)
    out |= {"r": r, "label": label, "pred": pred, "probs": probs, "logits": logit, "rr": rr, "searchback": sb, "ann_missed": missed,
            "z_q": ib["z_q"], "rr_parts": ib["rr_parts"], "beat_index": bi}
    return out
