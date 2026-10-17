"""Paired-by-seed comparison of pooled-OOF macro-F1 (4 classes) between a base run and candidates.

A candidate is a run directory, optionally with a fixed-epoch snapshot: DIR or DIR@<epoch>.
Decision rule (python/models/metrics.paired_by_seed): adopt only if the candidate improves
the pooled-OOF macro-F1 for every seed AND the mean improvement is >= 0.01. Also prints macro-F1 N/S/V, EC57 SVEB/VEB and N->F.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from python.models.metrics import paired_by_seed  # noqa: E402


def load(spec: str) -> tuple[str, list[dict]]:
    """DIR (final model), DIR@<epoch> (fixed-epoch snapshot) or DIR@best (snapshot with the highest
    mean pooled-OOF macro-F1 over seeds)."""
    path, _, epoch = spec.partition("@")
    m = json.loads((Path(path) / "metrics.json").read_text(encoding="utf-8"))
    if epoch == "best":
        epoch = max(m["snapshots"], key=lambda e: m["snapshots"][e]["mean_std"]["macro_f1"]["mean"])
    reps = m["snapshots"][epoch]["seed_reports"] if epoch else m["seed_reports"]
    return f"{Path(path).name}{'@' + epoch if epoch else ''}", reps


def row(name: str, reps: list[dict]) -> str:
    def ms(key):
        v = [key(r) for r in reps]
        mean = sum(v) / len(v)
        sd = (sum((x - mean) ** 2 for x in v) / (len(v) - 1)) ** 0.5 if len(v) > 1 else 0.0
        return f"{mean:.4f}±{sd:.4f}"
    return (f"{name:<48} {ms(lambda r: r['macro_f1'])}  {ms(lambda r: r['macro_f1_nsv'])}  "
            f"{ms(lambda r: r['per_class']['S']['recall'])}  "
            f"{ms(lambda r: r['ec57']['SVEB']['se'])}/{ms(lambda r: r['ec57']['SVEB']['ppv'])}  "
            f"{ms(lambda r: r['ec57']['VEB']['se'])}/{ms(lambda r: r['ec57']['VEB']['ppv'])}  "
            f"{sum(r['confusion'][0][3] for r in reps) / len(reps):7.0f}  "
            f"{sum(r['confusion'][0][2] for r in reps) / len(reps):7.0f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("candidates", nargs="+")
    args = ap.parse_args()
    bname, base = load(args.base)
    print(f"{'run':<48} {'macro-F1':>13}  {'mF1 N/S/V':>13}  {'recall S':>13}  {'SVEB Se / +P':>27}  {'VEB Se / +P':>27}  {'N->F':>7}  {'N->V':>7}")
    print(row(bname + " (base)", base))
    out = {}
    for spec in args.candidates:
        name, reps = load(spec)
        print(row(name, reps))
        out[name] = paired_by_seed([r["macro_f1"] for r in base], [r["macro_f1"] for r in reps])
    print("\npaired by seed (candidate - base, pooled-OOF macro-F1; adopt only if > 0 for every seed AND mean >= 0.01):")
    for name, p in out.items():
        print(f"  {name:<48} deltas {[round(x, 4) for x in p['delta']]}  mean {p['mean']:+.4f} ± {p['std']:.4f}"
              f"  all>0 {p['all_seeds_positive']}  mean>=0.01 {p['mean_ge_min']}"
              f"  -> {'ADOPT' if p['adopt'] else 'keep base'}")
    adopted = {n: p for n, p in out.items() if p["adopt"]}
    if adopted:
        best = max(adopted, key=lambda n: adopted[n]["mean"])
        print(f"\nSELECTED: {best} (highest mean delta among adopted candidates)")
    else:
        print("\nSELECTED: base (no candidate satisfies the rule)")


if __name__ == "__main__":
    main()
