"""Compare two eval_detector.py result directories (DS1).

Prints total Se/+P for both runs, the records whose Se or +P changed by more
than --min-change (percentage points), and flags records whose Se drops by more
than --max-se-drop in run B. Exit code 2 if any record exceeds that drop.
"""

import argparse
import csv
import json
import sys
from pathlib import Path


def _load(d: Path):
    metrics = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
    with open(d / "per_record.csv", encoding="utf-8") as f:
        rows = {int(r["rec"]): r for r in csv.DictReader(f)}
    return metrics, rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_a")
    ap.add_argument("run_b")
    ap.add_argument("--tol", default="150ms")
    ap.add_argument("--min-change", type=float, default=0.1)
    ap.add_argument("--max-se-drop", type=float, default=0.5)
    args = ap.parse_args()
    (ma, ra), (mb, rb) = _load(Path(args.run_a)), _load(Path(args.run_b))
    k = args.tol

    for name, m in (("A", ma), ("B", mb)):
        t = m["totals"][k]
        print(f"{name}: Se {100 * t['se']:.3f}%  +P {100 * t['ppv']:.3f}%  (TP {t['tp']} FN {t['fn']} FP {t['fp']})")
    for name, m in (("A", ma), ("B", mb)):
        if "fn_by_class" in m:
            print(f"{name} FN by class: " + ", ".join(
                f"{c} {v['fn']}/{v['n_ref']} ({100 * v['miss_rate']:.2f}%)" for c, v in m["fn_by_class"].items()))

    print(f"\nrecords with |dSe| or |d+P| > {args.min_change} pp ({k}):")
    print(f"{'rec':>4} | {'Se A':>7} {'Se B':>7} {'dSe':>7} | {'+P A':>7} {'+P B':>7} {'d+P':>7} | FN A→B  FP A→B")
    flagged = []
    for rec in sorted(ra):
        a, b = ra[rec], rb[rec]
        se_a, se_b = 100 * float(a[f"se_{k}"]), 100 * float(b[f"se_{k}"])
        pp_a, pp_b = 100 * float(a[f"ppv_{k}"]), 100 * float(b[f"ppv_{k}"])
        if abs(se_b - se_a) > args.min_change or abs(pp_b - pp_a) > args.min_change:
            print(f"{rec:>4} | {se_a:7.3f} {se_b:7.3f} {se_b - se_a:+7.3f} | {pp_a:7.3f} {pp_b:7.3f} "
                  f"{pp_b - pp_a:+7.3f} | {a[f'fn_{k}']:>3}→{b[f'fn_{k}']:<3} {a[f'fp_{k}']:>3}→{b[f'fp_{k}']:<3}")
        if se_a - se_b > args.max_se_drop:
            flagged.append((rec, se_a - se_b))
    if flagged:
        print(f"\nSTOP: Se drop > {args.max_se_drop} pp in run B: "
              + ", ".join(f"{r} (-{d:.3f} pp)" for r, d in flagged))
        sys.exit(2)
    print(f"\nno record drops Se by more than {args.max_se_drop} pp")


if __name__ == "__main__":
    main()
