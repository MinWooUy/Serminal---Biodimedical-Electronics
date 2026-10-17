"""Recompute metrics.json of a finished CV run from its saved OOF files (oof_s<seed>.npz),
with the current python/models/metrics.py (e.g. after adding EC57 SVEB/VEB and N/S/V macro-F1).
Training results are not changed; only the derived metrics are rewritten.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from python.models.metrics import format_report, mean_std, report  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+")
    args = ap.parse_args()
    for d in map(Path, args.run_dirs):
        m = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
        seed_reports = []
        for f in sorted(d.glob("oof_s*.npz")):
            if "_e" in f.stem:                       # snapshot files are handled by their own run
                continue
            z = np.load(f)
            if np.any(z["fold"] < 0):
                raise SystemExit(f"{f}: incomplete OOF")
            pred = z["probs"].argmax(1)
            seed_reports.append(report(z["y"], pred))
            seed = int(f.stem.split("_s")[1])
            for k in np.unique(z["fold"]):
                key = f"s{seed}_f{k}"
                sel = z["fold"] == k
                m["fold_reports"][key] = m["fold_reports"].get(key, {}) | report(z["y"][sel], pred[sel])
            print(f"{d.name} seed {seed}:\n{format_report(seed_reports[-1])}")
        m["seed_reports"] = seed_reports
        m["mean_std"] = mean_std(seed_reports)
        m["recomputed_with"] = "python/models/metrics.py (EC57 SVEB/VEB, macro-F1 N/S/V)"
        (d / "metrics.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
        ms = m["mean_std"]
        print(f"{d.name}: macro-F1 {ms['macro_f1']['mean']:.4f} ± {ms['macro_f1']['std']:.4f}")


if __name__ == "__main__":
    main()
