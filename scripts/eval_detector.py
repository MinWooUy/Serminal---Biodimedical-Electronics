"""Evaluate the Pan-Tompkins detector on DS1 (never DS2).

Se/+P use refined R positions vs original annotations (VF segments excluded).
Writes results/<date>_<hash>/: config.yaml, metrics.json, per_record.csv, log.txt.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from python.data.constants import CLASSES, Q_LABEL  # noqa: E402
from python.data.mitbih import load_record  # noqa: E402
from python.data.splits import get_ds1_records  # noqa: E402
from python.dsp.evaluate_detector import (  # noqa: E402
    evaluate_record, pair_errors_ms, se_ppv, summarize_errors,
)
from python.dsp.filters import GROUP_DELAY_REF_HZ, group_delay_samples  # noqa: E402
from python.dsp.pan_tompkins import detect_offline, detector_from_config  # noqa: E402
from python.dsp.realign import half_window_samples, realign  # noqa: E402
from python.experiment import Tee, load_config, make_results_dir  # noqa: E402

CLASS_NAMES = {i: c for i, c in enumerate(CLASSES)} | {Q_LABEL: "Q"}


def error_hints(rec, ev, dets) -> dict:
    """Heuristics to explain a record's errors (used for the worst-record notes)."""
    arr, m = ev["arrays"], ev["match"]
    ann, det = arr["ann"], arr["det_refined"]
    aami = rec.aami[arr["ann_idx"]]
    fn_cls = {CLASS_NAMES[c]: int(n) for c, n in zip(*np.unique(aami[m.fn], return_counts=True))}
    fp = det[m.fp]
    k = np.searchsorted(ann, fp)
    nxt = np.where(k < len(ann), ann[np.minimum(k, len(ann) - 1)] - fp, 10**9) * 1000 / rec.fs
    prv = np.where(k > 0, fp - ann[np.maximum(k - 1, 0)], 10**9) * 1000 / rec.fs
    det_sb = np.array([d.searchback for d in dets])[arr["det_idx"]]
    fp_min = np.bincount((fp // (60 * rec.fs)).astype(int), minlength=31) if len(fp) else np.zeros(31, int)
    return {
        "fn_by_class": fn_cls,
        "fp_p_like": int(np.sum((nxt >= 100) & (nxt <= 300))),     # 100-300 ms before next beat
        "fp_t_like": int(np.sum((prv >= 200) & (prv <= 450))),     # 200-450 ms after previous beat
        "fp_searchback": int(det_sb[m.fp].sum()) if len(m.fp) else 0,
        "fp_top_minutes": [int(i) for i in np.argsort(fp_min)[::-1][:3] if fp_min[i] > 0],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/detector_pt.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if cfg.get("dataset") != "DS1":
        raise SystemExit("this script only evaluates DS1")
    pt, sos = detector_from_config(cfg)
    tols = tuple(float(t) for t in cfg["eval"]["tolerances_ms"])
    out_dir = make_results_dir(cfg)
    log = Tee(out_dir / "log.txt")

    gd = group_delay_samples(sos, GROUP_DELAY_REF_HZ, pt.fs)
    log(f"config: {args.config} -> {out_dir}")
    log(f"0.5-40 Hz group delay @ {GROUP_DELAY_REF_HZ:g} Hz: {gd:.3f} samples = {gd / pt.fs * 1000:.2f} ms"
        f" (compensation {round(gd)} samples)")
    log(f"PT delays: LP {pt.lp_delay} + HP {pt.hp_delay} = band-pass {pt.bp_delay} samples; "
        f"+ deriv {pt.deriv_delay} + MWI {(pt.mwi_len - 1) / 2} = {pt.bp_delay + pt.deriv_delay + (pt.mwi_len - 1) / 2} samples")

    rows, pooled, hints = [], {"coarse": [], "refined": [], "realigned": []}, {}
    tot = {f"{int(t)}ms": np.zeros(3, dtype=np.int64) for t in tols}
    edge_det = 0
    cls_ref = {name: 0 for name in CLASS_NAMES.values()}
    cls_fn = {name: 0 for name in CLASS_NAMES.values()}
    for rid in get_ds1_records():
        rec = load_record(rid)
        res = detect_offline(rec.adc, rec.adc_zero, rec.adc_gain, pt, sos)
        ann_re, _ = realign(rec.r_samples, res.x_filt, res.delay, half_window_samples(pt.fs), pt.realign_mode)
        ev = evaluate_record(rec.r_samples, res.r_refined, res.r_coarse, ann_re, rec.vf_segments, pt.fs, tols)
        arr, pairs = ev["arrays"], ev["match"].pairs
        pooled["coarse"].append(pair_errors_ms(arr["ann"], arr["det_coarse"], pairs, pt.fs))
        pooled["refined"].append(pair_errors_ms(arr["ann"], arr["det_refined"], pairs, pt.fs))
        pooled["realigned"].append(pair_errors_ms(arr["ann_realigned"], arr["det_refined"], pairs, pt.fs))
        edge_det += sum(d.edge for d in res.detections)
        aami = rec.aami[arr["ann_idx"]]                     # non-VF annotations
        fn_mask = np.zeros(len(aami), dtype=bool)
        fn_mask[ev["match"].fn] = True
        for lab, name in CLASS_NAMES.items():
            cls_ref[name] += int(np.sum(aami == lab))
            cls_fn[name] += int(np.sum(fn_mask & (aami == lab)))
        row = {"rec": rid, "n_ref": ev["n_ref"], "n_det": ev["n_det"],
               "searchback": sum(d.searchback for d in res.detections)}
        for t in tols:
            k = f"{int(t)}ms"
            e = ev[k]
            tot[k] += [e["tp"], e["fn"], e["fp"]]
            row |= {f"tp_{k}": e["tp"], f"fn_{k}": e["fn"], f"fp_{k}": e["fp"],
                    f"se_{k}": e["se"], f"ppv_{k}": e["ppv"]}
        for name, key in (("coarse", "err_coarse_vs_ann"), ("refined", "err_refined_vs_ann"),
                          ("realigned", "err_refined_vs_realigned")):
            row[f"err_{name}_mean_ms"] = ev[key]["mean_ms"]
            row[f"err_{name}_std_ms"] = ev[key]["std_ms"]
        rows.append(row)
        hints[rid] = error_hints(rec, ev, res.detections)

    k0 = f"{int(tols[0])}ms"
    log(f"\n{'rec':>4} {'n_ref':>5} {'TP':>5} {'FN':>4} {'FP':>4} {'Se%':>7} {'+P%':>7} | "
        f"{'Se75%':>7} {'+P75%':>7} | {'sb':>3} | err ms coarse / refined / vs realigned (mean±std)")
    for r in rows:
        log(f"{r['rec']:>4} {r['n_ref']:>5} {r['tp_' + k0]:>5} {r['fn_' + k0]:>4} {r['fp_' + k0]:>4} "
            f"{100 * r['se_' + k0]:7.3f} {100 * r['ppv_' + k0]:7.3f} | "
            f"{100 * r['se_75ms']:7.3f} {100 * r['ppv_75ms']:7.3f} | {r['searchback']:>3} | "
            f"{r['err_coarse_mean_ms']:6.1f}±{r['err_coarse_std_ms']:4.1f} / "
            f"{r['err_refined_mean_ms']:6.1f}±{r['err_refined_std_ms']:4.1f} / "
            f"{r['err_realigned_mean_ms']:5.1f}±{r['err_realigned_std_ms']:4.1f}")

    totals = {}
    for k, (tp, fn, fp) in tot.items():
        se, ppv = se_ppv(int(tp), int(fn), int(fp))
        totals[k] = {"tp": int(tp), "fn": int(fn), "fp": int(fp), "se": se, "ppv": ppv}
        log(f"TOTAL {k}: TP {tp} FN {fn} FP {fp}  Se {100 * se:.3f}%  +P {100 * ppv:.3f}%")
    errs = {k: summarize_errors(np.concatenate(v)) for k, v in pooled.items()}
    for k, e in errs.items():
        log(f"position error ({k}): mean {e['mean_ms']:.2f} ms, std {e['std_ms']:.2f} ms, "
            f"p95|e| {e['p95_abs_ms']:.2f} ms (n={e['n']})")
    log(f"detections whose refinement hit the window edge (kept coarse): {edge_det}")
    fn_by_class = {n: {"n_ref": cls_ref[n], "fn": cls_fn[n],
                       "miss_rate": cls_fn[n] / cls_ref[n] if cls_ref[n] else float("nan")}
                   for n in cls_ref}
    log(f"FN by AAMI class @ {k0} (VF excluded): " + ", ".join(
        f"{n} {v['fn']}/{v['n_ref']} ({100 * v['miss_rate']:.2f}%)" for n, v in fn_by_class.items()))

    gate = totals[k0]["se"] >= 0.995 and totals[k0]["ppv"] >= 0.995
    log(f"GATE 2 (Se and +P >= 99.5% @ {k0}): {'PASS' if gate else 'FAIL'}")

    worst = sorted(rows, key=lambda r: r["fn_" + k0] + r["fp_" + k0], reverse=True)[:3]
    log("\n3 worst records (FN+FP):")
    for r in worst:
        log(f"  {r['rec']}: FN {r['fn_' + k0]} FP {r['fp_' + k0]}  hints {hints[r['rec']]}")

    with open(out_dir / "per_record.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    metrics = {"totals": totals, "fn_by_class": fn_by_class,
               "position_errors": errs, "edge_hits_det": edge_det,
               "group_delay_10hz_samples": gd, "gate2_pass": bool(gate),
               "per_record": rows, "worst3": [r["rec"] for r in worst],
               "hints": {str(k): v for k, v in hints.items()}}
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    log(f"saved -> {out_dir}")
    log.close()


if __name__ == "__main__":
    main()
