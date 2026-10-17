"""Beat gallery (PNG): what the frozen fully integer Lite pipeline confuses (observation only).

    python scripts/beat_gallery.py --split ds1                   # DS1 out-of-fold (iii), seed 42 CV models
    python scripts/beat_gallery.py --split ds2 [--seg annotation] # final model; needs ECG_ALLOW_DS2=1 (user only)

Grid rows = true class, columns = predicted class (DS2 detector segmentation adds a row "FP detection ->
predicted X"). Each cell overlays up to --n randomly chosen beats (rng --sample-seed, default 0) of the
integer z-scored window (z_q / 16, what the int8 CNN sees) and the mean of ALL beats of the cell; the
title gives the cell count and the median RR ratio (rr_prev / rr_avg8).
DS1 uses the out-of-fold predictions of the CV run (pipeline_int_eval.py --save-preds), NOT the final
model (trained on all DS1, so in-sample).
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from python.data.mitbih import REPO_ROOT  # noqa: E402
from python.quant.pipeline_int import INPUT_EXP  # noqa: E402
from python.viz.common import CLASSES, FP  # noqa: E402

DEFAULT_CV_RUN = REPO_ROOT / "results" / "2026-10-15_a50fb3e1_A5_svdb_n25_lite_e3"
ROW_FP = 4                         # extra grid row for FP detections


def rr_ratio(parts: np.ndarray) -> np.ndarray:
    p = np.asarray(parts, dtype=np.float64)
    return p[:, 0] * p[:, 3] / p[:, 2]


def gallery_cells(true: np.ndarray, pred: np.ndarray, n_per_cell: int, seed: int) -> dict[tuple[int, int], dict]:
    """Per (row, col) cell: all member indices and a reproducible random sample of at most n_per_cell.
    Rows 0..3 = true N/S/V/F, row 4 = FP detections (true == FP); other labels (Q, VF) are skipped."""
    rng = np.random.default_rng(seed)
    row = np.where(true == FP, ROW_FP, true)
    cells = {}
    for r in range(5):
        for c in range(4):
            idx = np.flatnonzero((row == r) & (pred == c))
            sample = np.sort(rng.choice(idx, size=n_per_cell, replace=False)) if len(idx) > n_per_cell else idx
            cells[(r, c)] = {"idx": idx, "sample": sample}
    return cells


def draw(cells, z_q, ratio, title: str, path: Path, fp_row: bool, fs: int = 360):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = 5 if fp_row else 4
    t = (np.arange(z_q.shape[1]) - z_q.shape[1] // 2) / fs * 1000
    fig, axes = plt.subplots(rows, 4, figsize=(16, 3.0 * rows), sharex=True, sharey=True, squeeze=False)
    for r in range(rows):
        for c in range(4):
            ax = axes[r, c]
            cell = cells[(r, c)]
            n = len(cell["idx"])
            diag = r == c
            for i in cell["sample"]:
                ax.plot(t, z_q[i] / 2.0**INPUT_EXP, lw=0.6, alpha=0.45, color="#1f77b4" if diag else "#d62728")
            if n:
                ax.plot(t, z_q[cell["idx"]].mean(0) / 2.0**INPUT_EXP, lw=1.8, color="k")
                med = np.median(ratio[cell["idx"]])
                ax.set_title(f"n={n}  shown {len(cell['sample'])}  RR ratio med {med:.2f}", fontsize=9)
            else:
                ax.set_title("n=0", fontsize=9)
            ax.axvline(0, color="#999", lw=0.5)
            ax.set_facecolor("#f3f8f3" if diag else ("#fbf3f3" if n else "white"))
            if r == 0:
                ax.annotate(f"predicted {CLASSES[c]}", xy=(0.5, 1.18), xycoords="axes fraction", ha="center",
                            fontsize=12, weight="bold")
            if c == 0:
                ax.set_ylabel("FP detection" if r == ROW_FP else f"true {CLASSES[r]}", fontsize=12, weight="bold")
            if r == rows - 1:
                ax.set_xlabel("ms from R")
    axes[0, 0].set_ylim(-8.5, 8.5)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def load_ds1(run: Path, model_seed: int) -> dict:
    f = run / "pipeline_int_preds.npz"
    if not f.exists():
        raise SystemExit(f"{f} missing: run scripts/pipeline_int_eval.py --run {run} --save-preds first")
    d = np.load(f)
    return {"true": d["y"].astype(np.int64), "pred": d[f"pred_iii_s{model_seed}"].astype(np.int64),
            "z_q": d["z_q"], "ratio": rr_ratio(d["rr_parts"])}


def load_ds2(seg: str) -> dict:
    from python.data.constants import require_ds2_access
    from python.data.mitbih import load_record
    from python.data.splits import get_ds2_records
    from python.viz.common import load_frozen, record_view

    require_ds2_access()                         # observation only; the user sets ECG_ALLOW_DS2=1
    fz = load_frozen()
    true, pred, zq, parts = [], [], [], []
    for rid in get_ds2_records():
        v = record_view(load_record(rid), fz, seg)
        bi = v["beat_index"]
        true.append(v["label"][bi])
        pred.append(v["pred"][bi])
        zq.append(v["z_q"])
        parts.append(v["rr_parts"])
        print(f"  {rid}: {len(bi)} classified R positions")
    return {"true": np.concatenate(true), "pred": np.concatenate(pred), "z_q": np.concatenate(zq),
            "ratio": rr_ratio(np.concatenate(parts))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("ds1", "ds2"), required=True)
    ap.add_argument("--seg", choices=("detector", "annotation"), default="detector", help="DS2 only")
    ap.add_argument("--run", default=str(DEFAULT_CV_RUN), help="DS1: CV run with pipeline_int_preds.npz")
    ap.add_argument("--model-seed", type=int, default=42, help="DS1: which CV seed's OOF predictions")
    ap.add_argument("--n", type=int, default=16, help="beats drawn per cell")
    ap.add_argument("--sample-seed", type=int, default=0)
    ap.add_argument("--out", default=str(REPO_ROOT / "viz"))
    args = ap.parse_args()
    if args.split == "ds1":
        g = load_ds1(Path(args.run), args.model_seed)
        title = f"DS1 out-of-fold, fully integer pipeline (iii), CV seed {args.model_seed} — rows true, cols predicted"
        name = f"gallery_ds1_oof_s{args.model_seed}.png"
        fp_row = False
    else:
        g = load_ds2(args.seg)
        title = f"DS2 (observation only), frozen final Lite, fully integer, {args.seg} segmentation"
        name = f"gallery_ds2_{args.seg}.png"
        fp_row = args.seg == "detector"
    cells = gallery_cells(g["true"], g["pred"], args.n, args.sample_seed)
    cm = np.array([[len(cells[(r, c)]["idx"]) for c in range(4)] for r in range(5 if fp_row else 4)])
    print("cells (rows true N/S/V/F" + (", FP" if fp_row else "") + "; cols predicted N/S/V/F):")
    for r in cm:
        print("  " + " ".join(f"{v:>7}" for v in r))
    path = Path(args.out) / name
    draw(cells, g["z_q"], g["ratio"], title, path, fp_row)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
