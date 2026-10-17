"""Baseline: logistic regression (class_weight='balanced') on the 4 RR features only.

Same record-grouped 5 folds as the CNN; trained on all train-fold beats (no early
stopping needed); macro-F1 on pooled out-of-fold predictions. Run for the same
seeds as the CNN (lbfgs is deterministic, so the std is expected to be 0).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from python.experiment import Tee, make_results_dir  # noqa: E402
from python.models.cv_data import load_cv_data  # noqa: E402
from python.models.metrics import CLASSES, format_report, mean_std, report  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--beats-config", default="configs/beats.yaml")
    ap.add_argument("--seeds", default="42,43,44")
    args = ap.parse_args()
    cv = load_cv_data(args.beats_config)
    d = cv.data
    cfg = {"name": "baseline_rr", "dataset": "DS1", "beats_config": args.beats_config,
           "model": "StandardScaler + LogisticRegression(class_weight=balanced, max_iter=2000)",
           "_cache": cv.cache.name}
    out = make_results_dir(cfg, suffix="_baseline_rr")
    log = Tee(out / "log.txt")
    reports = []
    for seed in [int(s) for s in args.seeds.split(",")]:
        pred = np.full(len(d["y"]), -1, dtype=np.int64)
        for k, recs in enumerate(cv.folds):
            val = np.isin(d["rec_id"], recs)
            clf = make_pipeline(StandardScaler(),
                                LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed))
            clf.fit(d["rr"][~val], d["y"][~val])
            pred[val] = clf.predict(d["rr"][val])
            fr = report(d["y"][val], pred[val])
            log(f"[seed {seed} fold {k}] macro-F1 {fr['macro_f1']:.4f}  recall " +
                ", ".join(f"{c} {fr['per_class'][c]['recall']:.3f}" for c in CLASSES))
        assert np.all(pred >= 0)
        np.savez_compressed(out / f"oof_pred_s{seed}.npz", pred=pred, y=d["y"], rec_id=d["rec_id"])
        rep = report(d["y"], pred)
        reports.append(rep)
        log(f"=== seed {seed}: pooled OOF ===\n" + format_report(rep))
    ms = mean_std(reports)
    log(f"\n=== baseline RR: OOF macro-F1 {ms['macro_f1']['mean']:.4f} ± {ms['macro_f1']['std']:.4f} ===")
    for c in CLASSES:
        log(f"  {c}: " + "  ".join(f"{m} {ms[c][m]['mean']:.4f}±{ms[c][m]['std']:.4f}"
                                   for m in ("precision", "recall", "f1")))
    (out / "metrics.json").write_text(json.dumps({"seed_reports": reports, "mean_std": ms}, indent=2),
                                      encoding="utf-8")
    log(f"saved -> {out}")
    log.close()


if __name__ == "__main__":
    main()
