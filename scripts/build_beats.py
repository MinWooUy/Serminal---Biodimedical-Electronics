"""Build the per-record beat cache (DS1 only) for one or both R sources and print gate-3 tables.

Cache: data/cache/beats_<source>_<hash>/<rec>.npz (+ <rec>.json stats).
Hash covers the beats config, the detector config and BEATS_VERSION.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from python.data.beats import (  # noqa: E402
    KIND_FP, KIND_LABELED, RR_FEATURES, RR_FEATURES_EXT, RR_SOURCES, BeatConfig, assert_ds1_only,
    build_record, cache_config, cache_dir, load_beats, load_stats, save_record,
)
from python.data.mitbih import REPO_ROOT, load_record  # noqa: E402
from python.data.splits import get_ds1_records, load_folds  # noqa: E402
from python.dsp.pan_tompkins import detector_from_config  # noqa: E402
from python.experiment import Tee, config_hash, load_config, make_results_dir  # noqa: E402

CLS = ("N", "S", "V", "F", "Q")


def build_source(cfg: dict, det_cfg: dict, source: str, records, log, force: bool) -> Path:
    full = cache_config(cfg, det_cfg, source)
    out = cache_dir(config_hash(full), source)
    if not force and all((out / f"{r}.npz").exists() for r in records):
        log(f"[{source}] cache exists: {out}")
        return out
    pt, sos = detector_from_config(det_cfg)
    bc = BeatConfig.from_dict(full["beats"])
    t0 = time.time()
    for rid in records:
        arrays, stats = build_record(load_record(rid), pt, sos, bc)
        save_record(out, rid, arrays, stats)
    (out / "config.json").write_text(json.dumps(full, indent=2), encoding="utf-8")
    size = sum(f.stat().st_size for f in out.glob("*.npz")) / 2**20
    log(f"[{source}] built {len(records)} records in {time.time() - t0:.1f} s, {size:.1f} MiB -> {out}")
    return out


def cfg_clip(out: Path) -> float:
    return float(json.loads((out / "config.json").read_text(encoding="utf-8"))["beats"]["beat_norm"]["clip"])


def report(out: Path, source: str, records, folds, log) -> dict:
    stats = load_stats(out, records)
    allb = load_beats(out, records, trainable_only=False)
    lab = allb["kind"] == KIND_LABELED
    tot = lambda key: {c: sum(s[key][c] for s in stats.values()) for c in CLS}  # noqa: E731
    n_ann, fn, fl, edge = tot("n_ann"), tot("fn"), tot("lost_first_last"), tot("lost_edge")
    final = {c: int(np.sum(lab & (allb["y"] == i))) for i, c in enumerate(CLS[:4])}
    final["Q"] = int(np.sum(lab & (allb["y"] == -1)))
    log(f"\n[{source}] beats per class: annotations -> lost (FN / first-last / window edge) -> dataset")
    log(f"{'cls':>3} {'ann':>6} {'FN':>5} {'f/l':>4} {'edge':>4} {'kept':>6} {'lost%':>7}")
    for c in CLS:
        kept = final[c] if c != "Q" else 0
        note = "  (Q excluded after RR)" if c == "Q" else ""
        lost = n_ann[c] - kept
        log(f"{c:>3} {n_ann[c]:>6} {fn[c]:>5} {fl[c]:>4} {edge[c]:>4} {kept:>6} "
            f"{100 * lost / max(n_ann[c], 1):6.2f}%{note}")
    n_fp = sum(s["fp"] for s in stats.values())
    log(f"FP detections: {n_fp} (kept in cache as kind=FP after first/last/edge drop: "
        f"{int(np.sum(allb['kind'] == KIND_FP))}); detections in VF: {sum(s['vf_det'] for s in stats.values())}")

    log(f"\n[{source}] train/val beats per fold (matched, non-Q)")
    log(f"{'fold':>4} {'N':>6} {'S':>5} {'V':>5} {'F':>5}")
    per_fold = []
    for i, recs in enumerate(folds):
        d = load_beats(out, recs)
        cnt = [int(np.sum(d["y"] == k)) for k in range(4)]
        per_fold.append(cnt)
        log(f"{i:>4} {cnt[0]:>6} {cnt[1]:>5} {cnt[2]:>5} {cnt[3]:>5}")

    d = load_beats(out, records)
    names = RR_FEATURES_EXT if d["rr"].shape[1] == len(RR_FEATURES_EXT) else RR_FEATURES
    log(f"\n[{source}] RR features per class (mean ± std): " + ", ".join(names))
    for k, c in enumerate(CLS[:4]):
        rr = d["rr"][d["y"] == k]
        log(f"{c:>3} " + "  ".join(f"{m:6.3f}±{s:5.3f}" for m, s in zip(rr.mean(0), rr.std(0))))
    log(f"[{source}] RR min / max (s) and mean rr_ratio per class")
    for k, c in enumerate(CLS[:4]):
        rr = d["rr"][d["y"] == k]
        log(f"{c:>3} rr_prev {rr[:, 0].min():.3f}/{rr[:, 0].max():.3f}  rr_next {rr[:, 1].min():.3f}/"
            f"{rr[:, 1].max():.3f}  rr_avg8 {rr[:, 2].min():.3f}/{rr[:, 2].max():.3f}  "
            f"ratio mean {rr[:, 3].mean():.3f} (min {rr[:, 3].min():.3f}, max {rr[:, 3].max():.3f})")
    clip = cfg_clip(out)
    beat_clipped = np.any(np.abs(d["X"]) >= clip, axis=1)
    log(f"X: shape {d['X'].shape}, mean {d['X'].mean():.3f}, std {d['X'].std():.3f}; beats touching "
        f"clip ±{clip:g}: " + ", ".join(f"{c} {100 * beat_clipped[d['y'] == k].mean():.2f}%"
                                        for k, c in enumerate(CLS[:4])))
    return {"n_ann": n_ann, "fn": fn, "first_last": fl, "edge": edge, "final": final,
            "fp": n_fp, "per_fold": per_fold}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/beats.yaml")
    ap.add_argument("--sources", default="detector,annotation")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if cfg.get("dataset") != "DS1":
        raise SystemExit("this script only builds DS1")
    det_cfg = load_config(REPO_ROOT / cfg["detector_config"])
    records = get_ds1_records()
    assert_ds1_only(records)
    folds = load_folds()
    out_dir = make_results_dir(cfg, suffix="_beats")
    log = Tee(out_dir / "log.txt")
    summary = {}
    for source in args.sources.split(","):
        if source not in RR_SOURCES:
            raise SystemExit(f"unknown source {source}")
        cdir = build_source(cfg, det_cfg, source, records, log, args.force)
        summary[source] = report(cdir, source, records, folds, log) | {"cache": str(cdir)}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(f"\nsaved -> {out_dir}")
    log.close()


if __name__ == "__main__":
    main()
