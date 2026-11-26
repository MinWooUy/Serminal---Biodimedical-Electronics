"""C1: logit-bias calibration for the frozen Lite, evaluated with NESTED cross-validation.

Adds (0, b_S, 0, b_F) to the logits before argmax (log-softmax of the OOF probabilities: same argmax as
the logits). Grid b_S, b_F in [-3, +1], step 0.25.
Nested: for each seed and outer fold k, (b_S, b_F) maximises pooled macro-F1 on the OOF of the other 4
folds (ties -> smaller |b_S| + |b_F|) and is applied to fold k; the pooled result is an honest estimate.
Paired-by-seed rule vs no calibration. If adopted: final (b_S, b_F) chosen on all 5 folds pooled over the
3 seeds; int8 check with the same b on the int8-path OOF logits (scripts/quant_eval.py output).
Export note: b is folded into the FC2 bias at quantisation (b_q = round(b * 2^(k2 + a5))).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from python.experiment import Tee  # noqa: E402
from python.models.metrics import format_report, mean_std, paired_by_seed, report  # noqa: E402

GRID = np.round(np.arange(-3.0, 1.0 + 1e-9, 0.25), 2)
S, F = 1, 3


def macro_f1_fast(y: np.ndarray, pred: np.ndarray) -> float:
    cm = np.bincount(y * 4 + pred, minlength=16).reshape(4, 4).astype(float)
    tp = np.diag(cm)
    denom = cm.sum(0) + cm.sum(1)
    f1 = np.where(denom > 0, 2 * tp / np.maximum(denom, 1e-12), 0.0)
    return float(f1.mean())


def apply_bias(lp: np.ndarray, b_s: float, b_f: float) -> np.ndarray:
    z = lp.copy()
    z[:, S] += b_s
    z[:, F] += b_f
    return z.argmax(1)


def best_bias(lp: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    best = None
    for b_s in GRID:
        for b_f in GRID:
            f1 = macro_f1_fast(y, apply_bias(lp, b_s, b_f))
            key = (f1, -(abs(b_s) + abs(b_f)))
            if best is None or key > best[0]:
                best = (key, float(b_s), float(b_f))
    return best[1], best[2], best[0][0]


def summary_line(rep: dict) -> str:
    cm = np.array(rep["confusion"])
    e = rep["ec57"]
    return (f"macro-F1 {rep['macro_f1']:.4f}  N/S/V {rep['macro_f1_nsv']:.4f}  SVEB Se/+P {e['SVEB']['se']:.3f}/"
            f"{e['SVEB']['ppv']:.3f}  VEB Se/+P {e['VEB']['se']:.3f}/{e['VEB']['ppv']:.3f}  N->F {cm[0, 3]}  N->V {cm[0, 2]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="CV run dir of the frozen Lite (oof_s<seed>.npz)")
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--int8-logits", default="quant_max_logits_s{seed}.npz")
    args = ap.parse_args()
    run = Path(args.run)
    seeds = [int(s) for s in args.seeds.split(",")]
    log = Tee(run / "c1_logit_bias.log")
    log(f"C1 logit-bias calibration on {run.name}; grid {GRID[0]}..{GRID[-1]} step 0.25 ({len(GRID) ** 2} pairs)")
    lps, ys, folds = {}, {}, {}
    for s in seeds:
        z = np.load(run / f"oof_s{s}.npz")
        lps[s] = np.log(np.clip(z["probs"], 1e-12, 1.0))
        ys[s], folds[s] = z["y"].astype(np.int64), z["fold"]
    rep_base, rep_cal, chosen = [], [], {}
    for s in seeds:
        lp, y, fo = lps[s], ys[s], folds[s]
        pred = np.empty(len(y), dtype=np.int64)
        for k in np.unique(fo):
            tr, te = fo != k, fo == k
            b_s, b_f, f1_in = best_bias(lp[tr], y[tr])
            pred[te] = apply_bias(lp[te], b_s, b_f)
            chosen[f"s{s}_f{k}"] = (b_s, b_f, f1_in)
            log(f"  seed {s} fold {k}: b_S {b_s:+.2f}  b_F {b_f:+.2f}  (inner 4-fold macro-F1 {f1_in:.4f})")
        rep_base.append(report(y, lp.argmax(1)))
        rep_cal.append(report(y, pred))
        log(f"seed {s} before: {summary_line(rep_base[-1])}")
        log(f"seed {s} nested: {summary_line(rep_cal[-1])}")
    p = paired_by_seed([r["macro_f1"] for r in rep_base], [r["macro_f1"] for r in rep_cal])
    mb, mc = mean_std(rep_base), mean_std(rep_cal)
    log(f"\nnested macro-F1: before {mb['macro_f1']['mean']:.4f} ± {mb['macro_f1']['std']:.4f} | after "
        f"{mc['macro_f1']['mean']:.4f} ± {mc['macro_f1']['std']:.4f}; deltas {[round(x, 4) for x in p['delta']]} "
        f"mean {p['mean']:+.4f}  all>0 {p['all_seeds_positive']}  mean>=0.01 {p['mean_ge_min']} -> "
        f"{'ADOPT' if p['adopt'] else 'keep (no calibration)'}")
    out = {"grid": GRID.tolist(), "chosen_per_fold": chosen, "before": rep_base, "nested": rep_cal, "paired": p}
    if p["adopt"]:
        lp_all = np.concatenate([lps[s] for s in seeds])
        y_all = np.concatenate([ys[s] for s in seeds])
        b_s, b_f, f1_all = best_bias(lp_all, y_all)
        log(f"\nFINAL (all 5 folds pooled over {len(seeds)} seeds): b_S {b_s:+.2f}  b_F {b_f:+.2f} "
            f"(in-sample pooled macro-F1 {f1_all:.4f}; the honest estimate is the nested one above)")
        out["final"] = {"b_S": b_s, "b_F": b_f}
        drops = []
        for s in seeds:
            q = run / args.int8_logits.format(seed=s)
            if not q.exists():
                log(f"  int8 check: {q.name} missing (run scripts/quant_eval.py first)")
                continue
            z = np.load(q)
            yy = z["y"].astype(np.int64)
            f_b = report(yy, apply_bias(z["logit_float"], b_s, b_f))
            q_b = report(yy, apply_bias(z["logit_int8"], b_s, b_f))
            drops.append(f_b["macro_f1"] - q_b["macro_f1"])
            log(f"  int8 check seed {s}: float+b {f_b['macro_f1']:.4f}  int8+b {q_b['macro_f1']:.4f}  "
                f"drop {drops[-1]:+.4f}")
        if drops:
            log(f"  int8 check: mean drop {np.mean(drops):+.4f} -> {'PASS' if np.mean(drops) <= 0.02 else 'FAIL'} (<= 0.02)")
            out["int8_drops"] = drops
    (run / "c1_logit_bias.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    log.close()


if __name__ == "__main__":
    main()
