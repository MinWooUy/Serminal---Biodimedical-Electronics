"""Classify a whole MIT-BIH record (or a time window of it) on the FPGA and score it against the annotations.

    python scripts/board_eval.py --record 208 --sim                          # no board: software FPGA model
    python scripts/board_eval.py --record 208 --serial COM5 --start 600 --dur 60   # short run on the board first
    python scripts/board_eval.py --record 208 --serial COM5                  # whole record (30 min of ECG)

PC side (everything except the CNN):
  1. load_record: MLII ADC samples + annotations (DS2 records raise unless the user set ECG_ALLOW_DS2=1).
  2. record_view (python/viz/common.py), the frozen fully integer pipeline: 0.5-40 Hz integer filter,
     R positions (--seg detector: integer Pan-Tompkins; annotation: the annotated R), R refinement,
     360-sample window, integer z-score -> z_q, RR parts -> rr_q; true labels (detector: one-to-one match
     with the annotations within 150 ms, FP / VF / missed annotations kept apart); the software int8 CNN
     (int_forward_q) on the same z_q / rr_q gives the reference logits.
  3. Every beat whose R lies in [--start, --start + --dur) is sent to the FPGA (host/board_classifier.py):
     WRITE_X + SWAP, then per layer WRITE_W, SWAP, WRITE_BIAS, WRITE_SHIFT, CONFIG, RUN; READ_FMAP 0,17
     returns 4 int32 logits + the argmax byte.  A failed beat is resent from scratch --retries times.
     With --seg both, a beat whose (z_q, rr_q) was already classified is not sent again (same input bytes
     -> same answer); the CSV marks it `reused`.
  4. Scores: FPGA logits vs software logits (must be bit-exact), FPGA labels vs annotations
     (P/R/F1 per class, macro-F1, confusion matrix; detector: also missed beats and FP detections).
FPGA side: runs the six layers of each beat (MAC, bias, shift, ReLU, MaxPool / GAP, FC1 row_combine, FC2
argmax) and answers every command with ACK / NACK / RESULT; it keeps nothing between beats.

Output: results/<date>_<hash>_board_eval/ with config.yaml, beats_<seg>.csv, metrics.json, log.txt and
(when plotly is installed) record_<id>_<seg>_board.html (view_record figure coloured by the FPGA labels).
DS1 records are IN-SAMPLE for the final model: the scores confirm FPGA == software, not model quality.
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from host.board_classifier import BoardError  # noqa: E402
from python.data.constants import DS1  # noqa: E402
from python.data.mitbih import load_record  # noqa: E402
from python.experiment import Tee, make_results_dir  # noqa: E402
from python.models.metrics import format_report, report  # noqa: E402
from python.quant.pipeline_int import quantise_rr_parts  # noqa: E402
from python.viz.common import CLASSES, FP, NOT_CLASSIFIED, VF, load_frozen, record_view, softmax_from_int_logits  # noqa: E402
from scripts.board_monitor import open_board  # noqa: E402
from scripts.final_eval_ds2 import gross_metrics  # noqa: E402

NAME = {0: "N", 1: "S", 2: "V", 3: "F", -1: "Q", FP: "FP", VF: "VF"}
CSV_FIELDS = ("seg", "beat", "r", "t_s", "true", "sw_pred", "board_pred", "board_eq_sw", "reused", "attempts",
              "cycles", "link_s", "logit_N", "logit_S", "logit_V", "logit_F", "sw_logit_N", "sw_logit_S",
              "sw_logit_V", "sw_logit_F", "error")


def classify_with_retries(clf, z_q, rr_q, retries: int):
    """-> (BoardResult | None, attempts, last error text)."""
    err = ""
    for attempt in range(1, retries + 2):
        try:
            return clf.classify(z_q, rr_q), attempt, ""
        except BoardError as exc:
            err = str(exc)
            clf.recover()
    return None, retries + 1, err


def run_segmentation(seg, v, window, clf, cache, retries, log, progress_every=50):
    """Send every classified beat of v with R in window to the board. Returns per-beat rows + arrays."""
    fs = v["fs"]
    a, b = window
    bi = v["beat_index"]                                  # R index of each row of z_q / rr_parts
    sel = np.flatnonzero((v["r"][bi] >= a) & (v["r"][bi] < b))
    rows, n = [], len(sel)
    board_pred = np.full(len(v["r"]), NOT_CLASSIFIED, dtype=np.int64)
    board_logit = np.zeros((len(v["r"]), 4), dtype=np.int64)
    t0 = time.perf_counter()
    for k, j in enumerate(sel):
        i = int(bi[j])
        z_q = v["z_q"][j]
        rr_q = v["rr_q"][j]
        key = z_q.astype(np.int8).tobytes() + rr_q.astype(np.int8).tobytes()
        reused = key in cache
        if reused:
            res, attempts, err = cache[key], 0, ""
        else:
            res, attempts, err = classify_with_retries(clf, z_q, rr_q, retries)
            if res is not None:
                cache[key] = res
        sw = [int(x) for x in v["logits"][i]]
        row = {"seg": seg, "beat": i, "r": int(v["r"][i]), "t_s": round(int(v["r"][i]) / fs, 3),
               "true": NAME.get(int(v["label"][i]), str(int(v["label"][i]))), "sw_pred": CLASSES[int(v["pred"][i])],
               "reused": int(reused), "attempts": attempts, "error": err,
               **{f"sw_logit_{c}": sw[m] for m, c in enumerate(CLASSES)}}
        if res is None:
            row |= {"board_pred": "", "board_eq_sw": 0, "cycles": "", "link_s": "",
                    **{f"logit_{c}": "" for c in CLASSES}}
            log(f"  [{seg}] beat {i} t={row['t_s']} s FAILED after {attempts} attempt(s): {err}")
        else:
            board_pred[i] = res.pred
            board_logit[i] = res.logits
            row |= {"board_pred": CLASSES[res.pred], "board_eq_sw": int(list(res.logits) == sw),
                    "cycles": res.cycles, "link_s": "" if reused else round(res.seconds, 4),
                    **{f"logit_{c}": int(res.logits[m]) for m, c in enumerate(CLASSES)}}
            if not row["board_eq_sw"]:
                log(f"  [{seg}] beat {i} t={row['t_s']} s: FPGA logits {list(res.logits)} != software {sw}")
        rows.append(row)
        if (k + 1) % progress_every == 0 or k + 1 == n:
            el = time.perf_counter() - t0
            eta = el / (k + 1) * (n - k - 1)
            log(f"  [{seg}] {k + 1}/{n} beats, {el:.0f} s elapsed, ETA {eta:.0f} s")
    return rows, board_pred, board_logit, sel


def score(seg, v, window, board_pred, sel_rows):
    """Metrics of the FPGA labels (and of the software labels on the same beats) against the annotations."""
    a, b = window
    lab, sw = v["label"], v["pred"]
    in_win = (v["r"] >= a) & (v["r"] < b)
    ok = in_win & (lab >= 0) & (board_pred >= 0)          # classified on the board, true class N/S/V/F
    out = {"beats_sent": int(len(sel_rows)),
           "beats_failed": int(sum(r["board_pred"] == "" for r in sel_rows)),
           "beats_reused": int(sum(r["reused"] for r in sel_rows)),
           "board_eq_sw": int(sum(r["board_eq_sw"] for r in sel_rows)),
           "board": report(lab[ok], board_pred[ok]), "software_same_beats": report(lab[ok], sw[ok])}
    if seg == "detector":
        ann = v["ann_pos"]
        missed = (ann >= a) & (ann < b) & v["ann_missed"]
        fn = np.array([int(np.sum(missed & (v["ann_label"] == c))) for c in range(4)])
        fp_pred = board_pred[in_win & (lab == FP) & (board_pred >= 0)]
        out["detector"] = {"fp_detections": int(np.sum(in_win & (lab == FP))), "missed_annotations": fn.tolist(),
                           "in_vf": int(np.sum(in_win & (lab == VF)))}
        out["board_gross"] = gross_metrics(lab[ok], board_pred[ok], fp_pred, fn)
    return out


def write_html(v, seg, rid, window, board_pred, board_logit, fz, out_dir, log):
    try:
        import plotly  # noqa: F401
    except ImportError:
        log("  HTML skipped: plotly is not installed in this Python (pip install plotly; it is in requirements.txt)")
        return None
    from scripts.view_record import build_figure
    w = dict(v)
    w["pred"] = board_pred
    probs = np.full((len(v["r"]), 4), np.nan)
    m = board_pred >= 0
    if m.any():
        probs[m] = softmax_from_int_logits(board_logit[m], fz.qm)
    w["probs"] = probs
    fs = v["fs"]
    t0, dur = window[0] / fs, (window[1] - window[0]) / fs
    fig = build_figure(w, rid, t0, dur, f"FPGA labels (frozen Lite int8 on the board), {seg} segmentation")
    path = out_dir / f"record_{rid}_{seg}_board.html"
    fig.write_html(path, include_plotlyjs=True)
    log(f"  HTML {path} ({path.stat().st_size / 1e6:.1f} MB)")
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--record", type=int, required=True)
    link = ap.add_mutually_exclusive_group(required=True)
    link.add_argument("--serial", help="serial port of the board, e.g. COM5 (needs pyserial)")
    link.add_argument("--sim", action="store_true", help="software FPGA model instead of a board")
    ap.add_argument("--seg", choices=("detector", "annotation", "both"), default="both")
    ap.add_argument("--start", type=float, default=0.0, help="window start (s)")
    ap.add_argument("--dur", type=float, default=None, help="window length (s), default to the end of the record")
    ap.add_argument("--baud", type=int, default=1500000, help="must match the bitstream")
    ap.add_argument("--timeout", type=float, default=3.0, help="per-command response timeout (s)")
    ap.add_argument("--retries", type=int, default=1, help="resend a failed beat from scratch this many times")
    ap.add_argument("--no-html", action="store_true", help="do not write the HTML view")
    args = ap.parse_args(argv)

    fz = load_frozen()
    rec = load_record(args.record)                       # DS2 -> error unless the user set ECG_ALLOW_DS2=1
    a = int(round(args.start * fz.fs))
    b = len(rec.adc) if args.dur is None else min(len(rec.adc), int(round((args.start + args.dur) * fz.fs)))
    if b <= a:
        raise SystemExit("empty window")
    segs = ("detector", "annotation") if args.seg == "both" else (args.seg,)
    cfg = {"name": "board_eval", "record": args.record, "segmentations": list(segs), "start_s": args.start,
           "dur_s": (b - a) / fz.fs, "link": "sim" if args.sim else args.serial, "baud": args.baud,
           "retries": args.retries, "lite": fz.lite_dir.name}
    out = make_results_dir(cfg, "_board_eval")
    log = Tee(out / "log.txt")
    note = "DS1: IN-SAMPLE (final model trained on all DS1)" if args.record in DS1 else "not DS1"
    log(f"record {args.record}, samples {a}..{b} ({(b - a) / fz.fs:.0f} s), segmentation {', '.join(segs)} -- {note}")

    transport, clf = open_board(args)
    hw = clf.hw
    log(f"FPGA INFO: {hw.rows}x{hw.cols} PE, {hw.clk_hz / 1e6:g} MHz, {hw.baud} baud, MAX_PAYLOAD {hw.max_payload}, "
        f"features 0x{hw.features:02X}" + ("" if hw.baud == args.baud else f"  WARNING: port opened at {args.baud}"))
    metrics = {"config": cfg, "fpga_info": {"rows": hw.rows, "cols": hw.cols, "clk_hz": hw.clk_hz, "baud": hw.baud}}
    cache, t_all = {}, time.perf_counter()
    try:
        for seg in segs:
            log(f"\n=== {seg} segmentation: PC pre-processing ===")
            v = record_view(rec, fz, seg)
            v["rr_q"] = quantise_rr_parts(v["rr_parts"], fz.qm.rr_exp, fz.fs)
            rows, board_pred, board_logit, sel = run_segmentation(seg, v, (a, b), clf, cache, args.retries, log)
            with open(out / f"beats_{seg}.csv", "w", newline="", encoding="utf-8") as f:
                wr = csv.DictWriter(f, fieldnames=CSV_FIELDS)
                wr.writeheader()
                wr.writerows(rows)
            m = score(seg, v, (a, b), board_pred, rows)
            metrics[seg] = m
            link_s = sum(r["link_s"] for r in rows if r["link_s"] != "")
            m["link_s_sent_beats"] = round(link_s, 2)
            log(f"\n--- {seg}: {m['beats_sent']} beats, failed {m['beats_failed']}, reused {m['beats_reused']}, "
                f"FPGA == software {m['board_eq_sw']}/{m['beats_sent'] - m['beats_failed']}; "
                f"link {link_s:.1f} s ({link_s / max(m['beats_sent'] - m['beats_reused'], 1):.2f} s/beat sent)")
            log(f"--- {seg}: FPGA labels vs annotations ---\n{format_report(m['board'])}")
            if m["software_same_beats"]["confusion"] != m["board"]["confusion"]:
                log(f"--- {seg}: software labels on the same beats (differs!) ---\n{format_report(m['software_same_beats'])}")
            if seg == "detector":
                d, g = m["detector"], m["board_gross"]
                log(f"--- detector: FP detections {d['fp_detections']}, missed annotations N/S/V/F {d['missed_annotations']}, "
                    f"in VF {d['in_vf']}; gross macro-F1 {g['macro_f1']:.4f}")
            if not args.no_html:
                write_html(v, seg, args.record, (a, b), board_pred, board_logit, fz, out, log)
    finally:
        transport.close()
        st = clf.host.stats
        metrics["link"] = {"transactions": st.transactions, "tx_bytes": st.tx_bytes, "rx_bytes": st.rx_bytes,
                           "wall_s": round(time.perf_counter() - t_all, 1), "unique_beats_classified": len(cache)}
        (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        log(f"\nlink: {st.transactions} transactions, {st.tx_bytes + st.rx_bytes} B, {len(cache)} distinct beats on "
            f"the board, {metrics['link']['wall_s']} s wall")
        log(f"results: {out}")
        log.close()
    ok = all(metrics[s]["beats_failed"] == 0 and metrics[s]["board_eq_sw"] == metrics[s]["beats_sent"] for s in segs)
    print("BOARD == SOFTWARE on every beat" if ok else "MISMATCH or failed beats: see log.txt / beats_*.csv")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
