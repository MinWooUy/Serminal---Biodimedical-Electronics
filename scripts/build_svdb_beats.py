"""Build the SVDB beat cache (training-only extra data, A5) and print SVDB vs DS1 beat counts.

Per record: channel rule A, then the unchanged MIT-BIH beat pipeline on the chosen channel
(detector R, one-to-one labels, RR, windows, z-score). Kept records -> <rec>.npz; manifest.json lists
kept/excluded records. Prints N/S/V/F train beats for SVDB 'all' and 'n25' (all S/V/F + 25% N) vs DS1.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from python.data.beats import BeatConfig, save_record  # noqa: E402
from python.data.mitbih import REPO_ROOT  # noqa: E402
from python.data.svdb import (  # noqa: E402
    DEFAULT_SVDB_DIR, SVDB_RULES, SVDB_RULES_NEG, available_records, build_svdb_negative_record, build_svdb_record,
    load_svdb_train, svdb_cache_dir, svdb_neg_cache_dir,
)
from python.dsp.pan_tompkins import detector_from_config  # noqa: E402
from python.experiment import Tee, load_config, make_results_dir  # noqa: E402
from python.models.cv_data import load_cv_data  # noqa: E402

CLS = ("N", "S", "V", "F")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--beats-config", default="configs/beats.yaml")
    ap.add_argument("--dir", default=str(DEFAULT_SVDB_DIR))
    ap.add_argument("--negative-fallback", action="store_true",
                    help="A5b: build ONLY the records excluded by rule A, on their negative channel if >= 97%%")
    args = ap.parse_args()
    bcfg = load_config(REPO_ROOT / args.beats_config)
    det_cfg = load_config(REPO_ROOT / bcfg["detector_config"])
    pt, sos = detector_from_config(det_cfg)
    bc = BeatConfig.from_dict({**bcfg["beats"], "rr_source": "detector"})
    main_cache = svdb_cache_dir(bcfg, det_cfg)
    if args.negative_fallback:
        out, rules, build = svdb_neg_cache_dir(bcfg, det_cfg), SVDB_RULES_NEG, build_svdb_negative_record
        records = sorted(int(r) for r in json.loads((main_cache / "manifest.json").read_text(encoding="utf-8"))["excluded"])
    else:
        out, rules, build = main_cache, SVDB_RULES, build_svdb_record
        records = available_records(args.dir)
    res = make_results_dir({"name": "svdb_beats", "beats": bcfg["beats"], "svdb": rules},
                           suffix="_svdb_neg_beats" if args.negative_fallback else "_svdb_beats")
    log = Tee(res / "log.txt")
    t0 = time.time()
    kept, excluded, stats_all = [], {}, []
    for rid in records:
        arrays, stats = build(rid, pt, sos, bc, args.dir)
        stats_all.append(stats)
        if arrays is None:
            excluded[rid] = stats["reason"]
            continue
        save_record(out, rid, arrays, {k: v for k, v in stats.items() if k != "channels"})
        kept.append(rid)
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps({"kept": kept, "excluded": excluded, "rules": rules,
                                                   "records": stats_all}, indent=2), encoding="utf-8")
    log(f"built SVDB cache {out} in {time.time() - t0:.0f} s: kept {len(kept)}/{len(records)}, "
        f"excluded {sorted(excluded)}")
    ds1 = load_cv_data(args.beats_config).data
    for r in stats_all:
        if r["chosen"] is not None:
            ch = r["channels"][r["chosen"]]
            log(f"  {r['rec']}: channel {r['chosen']} ({r['reason']}) Se {100 * ch['se']:.2f} +P {100 * ch['ppv']:.2f} "
                f"pos {ch['pos_frac']:.2f}")
    extra = (out,) if args.negative_fallback else ()
    rows = {"DS1 (MIT-BIH, train/val)": ds1["y"],
            "SVDB all" + (" + neg" if extra else ""): load_svdb_train(main_cache, "all", extra_caches=extra)["y"],
            "SVDB n25" + (" + neg" if extra else "") + " (seed 0)":
                load_svdb_train(main_cache, "n25", extra_caches=extra)["y"]}
    if extra:
        rows["  of which neg-fallback records (n25)"] = load_svdb_train(out, "n25")["y"]
    log(f"\n{'set':<40} " + " ".join(f"{c:>7}" for c in CLS) + f" {'total':>8}")
    for name, y in rows.items():
        log(f"{name:<40} " + " ".join(f"{int(np.sum(y == k)):>7}" for k in range(4)) + f" {len(y):>8}")
    for name in list(rows)[1:]:
        y = rows[name]
        log(f"{'DS1 + ' + name:<40} " + " ".join(f"{int(np.sum(ds1['y'] == k) + np.sum(y == k)):>7}"
                                                for k in range(4)) + f" {len(ds1['y']) + len(y):>8}")
    log(f"saved -> {res}")
    log.close()


if __name__ == "__main__":
    main()
