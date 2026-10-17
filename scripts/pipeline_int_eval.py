"""Integer-pipeline gate on DS1 OOF (frozen Lite, 15 fold x seed models).

For the cached train/val beats (detector R source) of every DS1 record:
  float pipeline : sosfilt -> float refinement -> z-score -> RR (s)          [cached X, rr]
  integer        : Q20/G12 DF1 filter -> int abs refinement -> int window -> int z-score -> int RR (python/quant/pipeline_int.py)
The integer detector (Pan-Tompkins) is shared: its coarse R positions are identical in both pipelines.
Cases per seed (pooled OOF, MIT-BIH beats only):
  float : float pre-processing + float (folded) model
  (i)   : integer pre-processing (dequantised) + float model
  (ii)  : float pre-processing + int8 model
  (iii) : integer pre-processing + int8 model  (fully integer)
GATE: macro-F1(float) - macro-F1(iii) <= 0.02, mean over seeds. Agreement / transitions are reported only.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from python.data.beats import CACHE_ROOT, KIND_LABELED  # noqa: E402
from python.data.mitbih import REPO_ROOT, load_record  # noqa: E402
from python.data.splits import get_ds1_records  # noqa: E402
from python.dsp.pan_tompkins import detect_offline, detector_from_config  # noqa: E402
from python.experiment import Tee, load_config  # noqa: E402
from python.models.cnn1d import ECGCNN, ModelConfig  # noqa: E402
from python.models.cv_data import attach_extra_train, load_cv_data, with_extra_train  # noqa: E402
from python.models.metrics import CLASSES, mean_std, report  # noqa: E402
from python.models.training import TrainConfig, fold_indices  # noqa: E402
from python.quant.int8 import float_forward, fold_model, int_forward, int_forward_q, quantize_model  # noqa: E402
from python.quant.pipeline_int import (  # noqa: E402
    FILTER_FRAC_BITS, FILTER_GUARD_BITS, INPUT_EXP, Q, filter_int, refine_abs_int, round_div, zscore_int,
)

MAX_DROP = 0.02


def integer_beats(cv, det_cfg: dict, beats_cfg: dict, log) -> dict[str, np.ndarray]:
    """Integer z_q windows and integer RR quantities (rp, rn, s8, n8) for every cached train/val beat,
    aligned with cv.data (same order)."""
    pt, sos = detector_from_config(det_cfg)
    half_win = int(beats_cfg["beats"]["half_window"])
    lo, hi = (int(round(v * pt.fs)) for v in beats_cfg["beats"]["rr_clip_s"])
    fcache = CACHE_ROOT / f"int_filter_F{FILTER_FRAC_BITS}_G{FILTER_GUARD_BITS}"
    fcache.mkdir(parents=True, exist_ok=True)
    d = cv.data
    z_q = np.zeros((len(d["y"]), 2 * half_win), dtype=np.int64)
    rrq = np.zeros((len(d["y"]), 4), dtype=np.int64)          # rp, rn, s8, n8 (samples)
    found = np.zeros(len(d["y"]), dtype=bool)
    for rid in get_ds1_records():
        rec = load_record(rid)
        f = fcache / f"{rid}.npy"
        if f.exists():
            y_int = np.load(f)
        else:
            y_int = filter_int(rec.adc, rec.adc_zero, sos)
            np.save(f, y_int)
        res = detect_offline(rec.adc, rec.adc_zero, rec.adc_gain, pt, sos)
        order = np.argsort(res.r_refined, kind="stable")                  # beats.r_sequence order
        r_float = res.r_refined[order]
        r_int = refine_abs_int(res.r_coarse[order], y_int, res.delay, 18)
        rr = np.clip(np.diff(r_int), lo, hi)
        sel = np.flatnonzero(d["rec_id"] == rid)
        pos = np.searchsorted(r_float, d["r_pos"][sel])
        if not np.array_equal(r_float[pos], d["r_pos"][sel]):
            raise RuntimeError(f"record {rid}: cached beats not found among detections")
        for j, i in zip(sel, pos):
            if i < 1 or i > len(r_int) - 2:
                raise RuntimeError("cached beat without previous/next R")
            c = int(r_int[i]) + res.delay
            if c - half_win < 0 or c + half_win > len(y_int):
                raise RuntimeError("integer window incomplete")
            past = rr[max(0, i - 8):i]
            rrq[j] = (rr[i - 1], rr[i], past.sum(), len(past))
            z_q[j] = zscore_int(y_int[c - half_win:c + half_win][None, :])[0]
            found[j] = True
        moved = int(np.sum(r_int != np.round(r_float).astype(np.int64)))
        log(f"  record {rid}: {len(sel)} beats; integer refinement differs from float for {moved}/{len(r_int)} detections")
    assert found.all()
    return {"z_q": z_q, "rr_parts": rrq}


def quantise_rr_parts(parts: np.ndarray, rr_exp: np.ndarray, fs: int = 360) -> np.ndarray:
    rp, rn, s8, n8 = (parts[:, k].astype(np.int64) for k in range(4))
    e = [int(v) for v in rr_exp]
    q = np.stack([round_div(rp << e[0], fs), round_div(rn << e[1], fs), round_div(s8 << e[2], fs * n8),
                  round_div((rp * n8) << e[3], s8)], axis=1)
    return np.clip(q, -Q, Q)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--n-calib", type=int, default=4000)
    ap.add_argument("--save-preds", action="store_true",
                    help="also save per-beat OOF (iii) logits / predictions per seed (pipeline_int_preds.npz)")
    args = ap.parse_args()
    run = Path(args.run)
    cfg = load_config(run / "config.yaml")
    bcfg = load_config(REPO_ROOT / cfg["beats_config"])
    det_cfg = load_config(REPO_ROOT / bcfg["detector_config"])
    cv = load_cv_data(cfg["beats_config"])
    tcfg = TrainConfig.from_dict(cfg["train"])
    log = Tee(run / "pipeline_int.log")
    log(f"integer pipeline gate on {run.name} (filter Q{FILTER_FRAC_BITS}, {FILTER_GUARD_BITS} guard bits)")
    ib = integer_beats(cv, det_cfg, bcfg, log)
    d, n_main, extra_idx, _ = attach_extra_train(cfg, cv.data)
    y = d["y"][:n_main]
    cases = ("float", "i", "ii", "iii")
    reps = {c: [] for c in cases}
    agree, trans, flips = [], [], []
    saved = {}
    for seed in cfg["seeds"]:
        pred = {c: np.full(n_main, -1) for c in cases}
        logits_iii = np.zeros((n_main, 4), dtype=np.int64)
        for k in cfg["folds"]:
            ck = torch.load(run / "ckpt" / f"s{seed}_f{k}.pt", weights_only=False)
            m = ECGCNN(ModelConfig.from_dict(ck["model"]))
            m.load_state_dict(ck["state_dict"])
            idx = fold_indices(d["rec_id"][:n_main], d["r_pos"][:n_main], cv.rec_len, cv.folds, k, cv.delay,
                               cv.half, tcfg, cv.fs)
            idx = with_extra_train(idx, extra_idx, n_main)
            rng = np.random.default_rng(seed * 10 + k)
            cal = np.sort(rng.choice(idx["train"], size=min(args.n_calib, len(idx["train"])), replace=False))
            fm = fold_model(m)
            qm = quantize_model(fm, d["X"][cal], d["rr"][cal], "max")
            out = idx["outer"]
            zq = ib["z_q"][out]
            rq = quantise_rr_parts(ib["rr_parts"][out], qm.rr_exp)
            pred["float"][out] = float_forward(fm, d["X"][out], d["rr"][out]).argmax(1)
            pred["i"][out] = float_forward(fm, zq / 2.0**INPUT_EXP, rq * 2.0 ** -qm.rr_exp.astype(float)).argmax(1)
            pred["ii"][out] = int_forward(qm, d["X"][out], d["rr"][out]).argmax(1)
            logits_iii[out] = int_forward_q(qm, zq, rq)
            pred["iii"][out] = logits_iii[out].argmax(1)
            if args.save_preds:
                saved.setdefault(f"logit_scale_s{seed}", np.zeros(n_main))[out] = 2.0 ** -(qm.fc2["k"] + qm.fc2["a_in"])
        if args.save_preds:
            saved[f"pred_iii_s{seed}"] = pred["iii"].astype(np.int8)
            saved[f"pred_float_s{seed}"] = pred["float"].astype(np.int8)
            saved[f"logits_iii_s{seed}"] = logits_iii.astype(np.int32)
        for c in cases:
            reps[c].append(report(y, pred[c]))
        agree.append({c: float(np.mean(pred[c] == pred["float"])) for c in cases[1:]})
        tr = np.zeros((4, 4), dtype=np.int64)
        np.add.at(tr, (pred["float"], pred["iii"]), 1)
        trans.append(tr.tolist())
        fl = pred["float"] != pred["iii"]
        flips.append({cn: int(np.sum(fl & (y == i))) for i, cn in enumerate(CLASSES)})
        log(f"\nseed {seed}: macro-F1 " + "  ".join(f"{c} {reps[c][-1]['macro_f1']:.4f}" for c in cases)
            + "  | argmax agreement with float: " + "  ".join(f"{c} {agree[-1][c]:.4f}" for c in cases[1:]))
        log(f"  float -> (iii) transitions (rows float N/S/V/F):\n" + "\n".join("    " + " ".join(f"{v:>6}" for v in r)
                                                                            for r in tr))
        log(f"  beats changed float -> (iii), by TRUE class: {flips[-1]} (total {int(fl.sum())})")
    drops = [f["macro_f1"] - q["macro_f1"] for f, q in zip(reps["float"], reps["iii"])]
    ms = {c: mean_std(reps[c]) for c in cases}
    log("\nmacro-F1 mean ± std: " + " | ".join(f"{c} {ms[c]['macro_f1']['mean']:.4f} ± {ms[c]['macro_f1']['std']:.4f}"
                                            for c in cases))
    for c in cases:
        e = ms[c]["ec57"]
        log(f"  {c:>5}: SVEB Se/+P {e['SVEB']['se']['mean']:.3f}/{e['SVEB']['ppv']['mean']:.3f}  "
            f"VEB Se/+P {e['VEB']['se']['mean']:.3f}/{e['VEB']['ppv']['mean']:.3f}")
    ok = float(np.mean(drops)) <= MAX_DROP
    log(f"INTEGER PIPELINE GATE: drop float -> (iii) per seed {[round(x, 4) for x in drops]}, mean "
        f"{np.mean(drops):+.4f} (<= {MAX_DROP}) -> {'PASS' if ok else 'FAIL'}")
    (run / "pipeline_int.json").write_text(json.dumps({"reports": reps, "agreement": agree, "transitions": trans,
                                                       "flips_by_true_class": flips, "drops": drops, "pass": ok},
                                                      indent=2), encoding="utf-8")
    if args.save_preds:
        np.savez_compressed(run / "pipeline_int_preds.npz", rec_id=d["rec_id"][:n_main], r_pos=d["r_pos"][:n_main], y=y,
                            z_q=ib["z_q"].astype(np.int8), rr_parts=ib["rr_parts"], seeds=np.array(cfg["seeds"]), **saved)
        log(f"saved per-beat OOF predictions -> {run / 'pipeline_int_preds.npz'}")
    log.close()


if __name__ == "__main__":
    main()
