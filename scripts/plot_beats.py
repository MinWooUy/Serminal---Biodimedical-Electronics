"""Gate 3 figure: 5 random train/val beats per class (N/S/V/F), R at sample 180, per R source.

Uses matplotlib (already installed as a wfdb dependency) with the Agg backend.
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from python.data.beats import cache_config, cache_dir, load_beats  # noqa: E402
from python.data.mitbih import REPO_ROOT  # noqa: E402
from python.data.splits import get_ds1_records  # noqa: E402
from python.experiment import config_hash, load_config, make_results_dir  # noqa: E402

CLASSES = ("N", "S", "V", "F")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/beats.yaml")
    ap.add_argument("--sources", default="detector,annotation")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cfg = load_config(args.config)
    det_cfg = load_config(REPO_ROOT / cfg["detector_config"])
    out_dir = make_results_dir(cfg, suffix="_beats")
    for source in args.sources.split(","):
        cdir = cache_dir(config_hash(cache_config(cfg, det_cfg, source)), source)
        d = load_beats(cdir, get_ds1_records())
        rng = np.random.default_rng(args.seed)
        fig, axes = plt.subplots(len(CLASSES), args.n, figsize=(3 * args.n, 2.2 * len(CLASSES)),
                                 sharex=True, sharey=True)
        for row, cls in enumerate(CLASSES):
            idx = rng.choice(np.flatnonzero(d["y"] == row), size=args.n, replace=False)
            for col, i in enumerate(idx):
                ax = axes[row, col]
                ax.plot(d["X"][i], lw=0.8, color="k")
                ax.axvline(180, color="r", lw=0.6, ls="--")
                ax.set_title(f"{cls}  rec {d['rec_id'][i]}  RRp {d['rr'][i, 0]:.2f}s", fontsize=8)
                ax.tick_params(labelsize=7)
            axes[row, 0].set_ylabel(f"{cls} (z)", fontsize=9)
        clip = cfg["beats"]["beat_norm"]["clip"]
        fig.suptitle(f"{source}: {args.n} beats/class, R at sample 180 (red), z-score clip ±{clip:g}", fontsize=10)
        fig.tight_layout()
        path = out_dir / f"beats_{source}.png"
        fig.savefig(path, dpi=110)
        plt.close(fig)
        print(f"saved {path}")


if __name__ == "__main__":
    main()
