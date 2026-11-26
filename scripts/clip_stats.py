"""Choose beat_norm.clip from the distribution of per-beat max |z| (unclipped z-score), DS1.

Rule: the smallest integer c such that, for every class N/S/V/F, at most 0.5% of the
train/val beats have max|z| > c (i.e. would be clipped). Selection uses the default
R source from the config; the other source is printed for reference.
"""

import argparse
import dataclasses
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from python.data.beats import KIND_LABELED, RR_SOURCES, BeatConfig, build_record  # noqa: E402
from python.data.mitbih import REPO_ROOT, load_record  # noqa: E402
from python.data.splits import get_ds1_records  # noqa: E402
from python.dsp.pan_tompkins import detector_from_config  # noqa: E402
from python.experiment import Tee, load_config, make_results_dir  # noqa: E402

CLASSES = ("N", "S", "V", "F")
THRESHOLDS = (5, 8, 10, 12, 15)
MAX_CLIPPED = 0.005


def per_beat_max_abs_z(source: str, cfg: dict, det_cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    pt, sos = detector_from_config(det_cfg)
    bc = dataclasses.replace(BeatConfig.from_dict({**cfg["beats"], "rr_source": source}),
                             norm_clip=math.inf)
    zmax, ys = [], []
    for rid in get_ds1_records():
        arr, _ = build_record(load_record(rid), pt, sos, bc)
        m = (arr["kind"] == KIND_LABELED) & (arr["y"] >= 0)
        zmax.append(np.abs(arr["X"][m]).max(axis=1))
        ys.append(arr["y"][m])
    return np.concatenate(zmax), np.concatenate(ys)


def clipped_frac(zmax, y, c) -> dict[str, float]:
    return {cls: float(np.mean(zmax[y == k] > c)) for k, cls in enumerate(CLASSES)}


def choose_clip(zmax, y) -> int:
    c = 1
    while max(clipped_frac(zmax, y, c).values()) > MAX_CLIPPED:
        c += 1
    return c


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/beats.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    det_cfg = load_config(REPO_ROOT / cfg["detector_config"])
    out_dir = make_results_dir(cfg, suffix="_clip")
    log = Tee(out_dir / "log.txt")
    default = cfg["beats"]["rr_source"]
    chosen = None
    for source in [default] + [s for s in RR_SOURCES if s != default]:
        zmax, y = per_beat_max_abs_z(source, cfg, det_cfg)
        c = choose_clip(zmax, y)
        if source == default:
            chosen = c
        log(f"\n[{source}] per-beat max|z| percentiles (50/95/99/99.5/max):")
        for k, cls in enumerate(CLASSES):
            p = np.percentile(zmax[y == k], [50, 95, 99, 99.5, 100])
            log(f"  {cls}: " + " / ".join(f"{v:5.2f}" for v in p) + f"   (n={int(np.sum(y == k))})")
        log(f"[{source}] % beats clipped (max|z| > c):")
        log(f"  {'c':>4} " + " ".join(f"{cls:>7}" for cls in CLASSES))
        for t in sorted(set(THRESHOLDS) | {c}):
            fr = clipped_frac(zmax, y, t)
            tag = "  <- smallest c with every class <= 0.5%" if t == c else ""
            log(f"  {t:>4} " + " ".join(f"{100 * fr[cls]:6.2f}%" for cls in CLASSES) + tag)
    log(f"\nCHOSEN beat_norm.clip = {chosen} (source: {default})")
    log.close()


if __name__ == "__main__":
    main()
