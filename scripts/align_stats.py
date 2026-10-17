"""Compare R re-alignment modes (signed / abs / none) on DS1, per AAMI class.

(a) shift of re-aligned annotation vs original annotation (mean, std, ms)
(b) re-aligned detection - re-aligned annotation over matched pairs (mean, std, p95|.|, ms)
    -> same position_errors() as eval_detector's "refined vs realigned"
(c) window-edge hits (annotations; detections matched to that class)

Matching = 150 ms one-to-one, original annotation vs re-aligned detection, VF excluded.
Selection rule: the mode with the smallest p95|(b)| for V and for F (ties broken
by the smaller std of (b)); if V and F prefer different modes, no mode is chosen
and the conflict is reported.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from python.data.constants import CLASSES, Q_LABEL  # noqa: E402
from python.data.mitbih import load_record  # noqa: E402
from python.data.splits import get_ds1_records  # noqa: E402
from python.dsp.evaluate_detector import (  # noqa: E402
    match_beats, outside_vf, pair_errors_ms, summarize_errors, tol_samples,
)
from python.dsp.pan_tompkins import detect_offline, detector_from_config  # noqa: E402
from python.dsp.realign import MODES, half_window_samples, realign  # noqa: E402
from python.experiment import Tee, load_config, make_results_dir  # noqa: E402

LABELS = list(enumerate(CLASSES)) + [(Q_LABEL, "Q")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/detector_pt.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if cfg.get("dataset") != "DS1":
        raise SystemExit("this script only evaluates DS1")
    pt, sos = detector_from_config(cfg)
    half = half_window_samples(pt.fs)
    tol = tol_samples(float(cfg["eval"]["tolerances_ms"][0]), pt.fs)
    out_dir = make_results_dir(cfg, suffix="_align")
    log = Tee(out_dir / "log.txt")

    acc = {m: {c: {"a": [], "b": [], "edge_ann": 0, "edge_det": 0, "rec_std": []} for _, c in LABELS}
           for m in MODES}
    for rid in get_ds1_records():
        rec = load_record(rid)
        res = detect_offline(rec.adc, rec.adc_zero, rec.adc_gain, pt, sos)   # coarse R is mode-independent
        ka = outside_vf(rec.r_samples, rec.vf_segments)
        ann, aami = rec.r_samples[ka], rec.aami[ka]
        coarse = res.r_coarse
        for mode in MODES:
            ann_re, edge_a = realign(ann, res.x_filt, res.delay, half, mode)
            det_re, edge_d = realign(coarse, res.x_filt, res.delay, half, mode)
            kd = outside_vf(det_re, rec.vf_segments)
            det_re, edge_d = det_re[kd], edge_d[kd]
            m = match_beats(ann, det_re, tol)
            pair_cls = aami[m.pairs[:, 0]]
            for lab, name in LABELS:
                a = acc[mode][name]
                sel = aami == lab
                shift = (ann_re[sel] - ann[sel]) * 1000.0 / pt.fs
                a["a"].append(shift)
                if len(shift) >= 2:                        # within-(record, class) spread: R/S jumping
                    a["rec_std"].append(float(shift.std()))
                a["edge_ann"] += int(edge_a[sel].sum())
                p = m.pairs[pair_cls == lab]
                a["b"].append(pair_errors_ms(ann_re, det_re, p, pt.fs))
                a["edge_det"] += int(edge_d[p[:, 1]].sum())

    summary = {}
    for mode in MODES:
        log(f"\n=== mode: {mode} ===")
        log(f"{'cls':>3} {'n':>6} | (a) shift mean±std ms | (b) det-ann mean±std, p95|.| ms | (c) edge ann / det")
        summary[mode] = {}
        for _, name in LABELS:
            a = acc[mode][name]
            sa = summarize_errors(np.concatenate(a["a"]))
            sb = summarize_errors(np.concatenate(a["b"]))
            summary[mode][name] = {"a": sa, "b": sb, "edge_ann": a["edge_ann"], "edge_det": a["edge_det"]}
            log(f"{name:>3} {sa['n']:>6} | {sa['mean_ms']:7.2f} ± {sa['std_ms']:6.2f}      | "
                f"{sb['mean_ms']:7.2f} ± {sb['std_ms']:6.2f}, p95 {sb['p95_abs_ms']:6.2f}      | "
                f"{a['edge_ann']:>5} / {a['edge_det']:>5}")

    # R/S jumping: std of the shift vs the original annotation within each (record, class),
    # averaged over records. Informational only: does not change the chosen mode.
    log("\nwithin-(record, class) std of shift vs original annotation, mean over records (ms)")
    log(f"{'cls':>3} | " + " | ".join(f"{md:>16}" for md in ("signed", "abs")) + " | n records")
    jump = {}
    for _, name in LABELS:
        jump[name] = {md: (float(np.mean(acc[md][name]["rec_std"])) if acc[md][name]["rec_std"] else float("nan"))
                      for md in ("signed", "abs")}
        log(f"{name:>3} | " + " | ".join(f"{jump[name][md]:16.2f}" for md in ("signed", "abs"))
            + f" | {len(acc['abs'][name]['rec_std'])}")
    for md in ("signed", "abs"):
        for _, name in LABELS:
            summary[md][name]["rec_std_mean_ms"] = jump[name][md]

    best = {}
    for name in ("V", "F"):
        # p95 is quantised to whole samples (2.78 ms) and often ties at 0: break ties by std.
        best[name] = min(MODES, key=lambda md: (summary[md][name]["b"]["p95_abs_ms"],
                                                 summary[md][name]["b"]["std_ms"]))
        log(f"best mode for {name} (min p95|b|, tie -> min std): {best[name]} "
            f"({ {md: (round(summary[md][name]['b']['p95_abs_ms'], 2), round(summary[md][name]['b']['std_ms'], 2)) for md in MODES} })")
    chosen = best["V"] if best["V"] == best["F"] else None
    log(f"CHOSEN: {chosen}" if chosen else f"CONFLICT: V -> {best['V']}, F -> {best['F']} (ask the user)")

    (out_dir / "metrics.json").write_text(
        json.dumps({"summary": summary, "best": best, "chosen": chosen}, indent=2), encoding="utf-8")
    log(f"saved -> {out_dir}")
    log.close()


if __name__ == "__main__":
    main()
