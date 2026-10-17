"""SVDB channel selection + detector Se/+P per record (A5 data gate).

For every downloaded SVDB record and both channels: resample to 360 Hz, x200 -> integers, run the
integer Pan-Tompkins, Se/+P (+-150 ms, VF excluded) and the fraction of positive QRS. Keep a record only
if it has a mostly-positive channel with Se and +P >= 97% (option A, python/data/svdb.py). Gate: total
Se and +P >= 99% over the kept records. Also reports the share of annotated S beats lost by the rule
(warn if > 15%). Writes results/<date>_<hash>_svdb_select/selection.json.
"""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from python.data.mitbih import REPO_ROOT  # noqa: E402
from python.data.svdb import (  # noqa: E402
    DEFAULT_SVDB_DIR, MIN_SE_PPV, available_records, choose_channel, evaluate_channel, load_channel, n_channels,
)
from python.dsp.pan_tompkins import detector_from_config  # noqa: E402
from python.experiment import Tee, load_config, make_results_dir  # noqa: E402

GATE = 0.99
MAX_S_LOSS = 0.15
CLS = {0: "N", 1: "S", 2: "V", 3: "F", -1: "Q"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--beats-config", default="configs/beats.yaml")
    ap.add_argument("--dir", default=str(DEFAULT_SVDB_DIR))
    args = ap.parse_args()
    det_cfg = load_config(REPO_ROOT / load_config(REPO_ROOT / args.beats_config)["detector_config"])
    pt, sos = detector_from_config(det_cfg)
    records = available_records(args.dir)
    out = make_results_dir({"name": "svdb_select", "rule": "A-strict", "detector": det_cfg,
                            "min_se_ppv": MIN_SE_PPV, "records": records}, suffix="_svdb_select")
    log = Tee(out / "log.txt")
    log(f"{len(records)} SVDB records available: {records}")
    log(f"{'rec':>4} | {'ch0 Se':>7} {'+P':>7} {'pos':>5} | {'ch1 Se':>7} {'+P':>7} {'pos':>5} | "
        f"{'chosen':>6} {'keep':>4} | {'n_S':>5} | reason")
    rows, tot = [], [0, 0, 0]
    for rid in records:
        recs = [load_channel(rid, ch, args.dir) for ch in range(n_channels(rid, args.dir))]
        evals = [evaluate_channel(r, pt, sos, ch) for ch, r in enumerate(recs)]
        best, reason = choose_channel(evals)
        counts = {CLS[k]: int(np.sum(recs[0].aami == k)) for k in CLS}
        if best is not None:
            tot[0] += best.tp
            tot[1] += best.fn
            tot[2] += best.fp
        cells = " | ".join(f"{e.se * 100:7.2f} {e.ppv * 100:7.2f} {e.pos_frac:5.2f}" for e in evals)
        log(f"{rid:>4} | {cells} | {best.ch if best else '-':>6} {'Y' if best else 'N':>4} | "
            f"{counts['S']:>5} | {reason}")
        rows.append({"rec": rid, "channels": [asdict(e) for e in evals], "chosen": best.ch if best else None,
                     "keep": best is not None, "reason": reason, "counts": counts,
                     "max_mwi_log2": max(e.max_mwi_log2 for e in evals)})
    tp, fn, fp = tot
    se = tp / (tp + fn) if tp + fn else float("nan")
    ppv = tp / (tp + fp) if tp + fp else float("nan")
    kept = [r["rec"] for r in rows if r["keep"]]
    s_all = sum(r["counts"]["S"] for r in rows)
    s_lost = sum(r["counts"]["S"] for r in rows if not r["keep"])
    log(f"\nkept {len(kept)}/{len(rows)} records; excluded: {[r['rec'] for r in rows if not r['keep']]}")
    log("annotated beats in kept records: " + ", ".join(
        f"{c} {sum(r['counts'][c] for r in rows if r['keep'])}" for c in ("N", "S", "V", "F", "Q")))
    log(f"S beats lost by the rule: {s_lost}/{s_all} = {100 * s_lost / max(s_all, 1):.1f}%"
        + ("  ** > 15%: report to the user **" if s_all and s_lost / s_all > MAX_S_LOSS else ""))
    log(f"total over kept: TP {tp} FN {fn} FP {fp}  Se {100 * se:.3f}%  +P {100 * ppv:.3f}%")
    log(f"SVDB detector gate (Se and +P >= {GATE:.0%} on kept records): "
        f"{'PASS' if se >= GATE and ppv >= GATE else 'FAIL'}")
    (out / "selection.json").write_text(json.dumps(
        {"records": rows, "kept": kept, "se": se, "ppv": ppv, "s_lost": s_lost, "s_all": s_all}, indent=2),
        encoding="utf-8")
    log(f"saved -> {out}")
    log.close()


if __name__ == "__main__":
    main()
