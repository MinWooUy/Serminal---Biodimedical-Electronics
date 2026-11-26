"""Gate 4 report from finished CV result directories (Full, Lite, RR baseline).

Prints mean ± std (over seeds) of pooled-OOF macro-F1 and per-class P/R/F1,
mean stop epoch, the pooled OOF confusion matrices, and checks the gate:
  - every class has recall > 0 (pooled OOF, every seed) for Full and Lite
  - Full macro-F1 (mean) > baseline macro-F1 (mean)
Plots the outer-fold macro-F1 per epoch (DIAGNOSTIC log, mean over folds x seeds)
for Full and Lite with the baseline as a horizontal line, and the confusion matrices.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from python.models.metrics import CLASSES  # noqa: E402


def load(d: Path) -> dict:
    return json.loads((d / "metrics.json").read_text(encoding="utf-8"))


def outer_curve(d: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean/std of the diagnostic outer macro-F1 per epoch over runs that reached that epoch."""
    runs = [[h["diag"]["outer_macro_f1"] for h in json.loads(p.read_text(encoding="utf-8"))]
            for p in sorted(d.glob("history_s*_f*.json"))]
    n = max(len(r) for r in runs)
    mean, std, cnt = np.full(n, np.nan), np.full(n, np.nan), np.zeros(n, dtype=int)
    for e in range(n):
        v = [r[e] for r in runs if len(r) > e]
        mean[e], std[e], cnt[e] = np.mean(v), np.std(v), len(v)
    return mean, std, cnt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", required=True)
    ap.add_argument("--lite", required=True)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dirs = {"Full": Path(args.full), "Lite": Path(args.lite), "Baseline RR": Path(args.baseline)}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    m = {k: load(v) for k, v in dirs.items()}
    lines = []

    def log(s=""):
        print(s)
        lines.append(s)

    log("pooled OOF over 5 folds; mean ± std over seeds")
    log(f"{'model':<12} {'macro-F1':>16} | " + " | ".join(f"{c}: P / R / F1" for c in CLASSES))
    for name, mm in m.items():
        ms = mm["mean_std"]
        cells = [" / ".join(f"{ms[c][x]['mean']:.3f}" for x in ("precision", "recall", "f1")) for c in CLASSES]
        log(f"{name:<12} {ms['macro_f1']['mean']:.4f} ± {ms['macro_f1']['std']:.4f} | " + " | ".join(cells))
    log("\nper-class std (P / R / F1):")
    for name, mm in m.items():
        ms = mm["mean_std"]
        log(f"{name:<12} " + " | ".join(
            f"{c}: " + "/".join(f"{ms[c][x]['std']:.3f}" for x in ("precision", "recall", "f1")) for c in CLASSES))

    log("\n(a) stop epoch (early stopping on inner-val loss):")
    for name in ("Full", "Lite"):
        s = m[name]["stop_epoch"]
        best = [v["best_epoch"] for v in m[name]["fold_reports"].values()]
        log(f"  {name}: stop {s['mean']:.1f} ± {s['std']:.1f} (min {s['min']}, max {s['max']}); "
            f"best epoch {np.mean(best):.1f} ± {np.std(best, ddof=1):.1f}")

    log("\n(c) pooled OOF confusion matrices (rows true N/S/V/F, cols predicted)")
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, name in zip(axes, ("Full", "Lite")):
        cms = np.array([r["confusion"] for r in m[name]["seed_reports"]])
        for s, cm in zip((42, 43, 44), cms):
            log(f"  {name} seed {s}: " + " ; ".join(" ".join(f"{v:>5}" for v in row) for row in cm))
        mean_cm = cms.mean(0)
        rown = mean_cm / mean_cm.sum(1, keepdims=True)
        ax.imshow(rown, cmap="Blues", vmin=0, vmax=1)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{mean_cm[i, j]:.0f}\n{100 * rown[i, j]:.1f}%", ha="center", va="center",
                        fontsize=8, color="white" if rown[i, j] > 0.6 else "black")
        ax.set_xticks(range(4), CLASSES)
        ax.set_yticks(range(4), CLASSES)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(f"{name}: pooled OOF, mean of 3 seeds", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "confusion_full_lite.png", dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for name, color in (("Full", "tab:blue"), ("Lite", "tab:orange")):
        mean, std, cnt = outer_curve(dirs[name])
        ep = np.arange(1, len(mean) + 1)
        ax.plot(ep, mean, color=color, label=f"{name} (mean over runs still training)")
        ax.fill_between(ep, mean - std, mean + std, color=color, alpha=0.15)
        ax.axvline(m[name]["stop_epoch"]["mean"], color=color, ls=":", lw=0.9)
    base = m["Baseline RR"]["mean_std"]["macro_f1"]["mean"]
    ax.axhline(base, color="k", ls="--", lw=0.9, label=f"baseline RR (pooled OOF) {base:.3f}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("outer-fold macro-F1 (per fold)")
    ax.set_title("DIAGNOSTIC: outer-fold macro-F1 per epoch (not used for training decisions)", fontsize=8)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "outer_f1_per_epoch.png", dpi=110)
    plt.close(fig)
    log("\n(b) saved outer_f1_per_epoch.png (dotted = mean stop epoch); note: per-fold macro-F1, "
        "not pooled, so not directly comparable to the pooled baseline line")

    rec_ok = {n: all(r["per_class"][c]["recall"] > 0 for r in m[n]["seed_reports"] for c in CLASSES)
              for n in ("Full", "Lite")}
    beats = m["Full"]["mean_std"]["macro_f1"]["mean"] > base
    log(f"\nGATE 4: recall > 0 for every class (every seed): Full {rec_ok['Full']}, Lite {rec_ok['Lite']}; "
        f"Full > baseline: {beats} ({m['Full']['mean_std']['macro_f1']['mean']:.4f} vs {base:.4f}) -> "
        f"{'PASS' if all(rec_ok.values()) and beats else 'FAIL'}")
    (out / "gate4.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
