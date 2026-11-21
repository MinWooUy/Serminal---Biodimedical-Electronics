"""Live side-by-side view of one MIT-BIH record: PhysioNet annotation | trained Lite (float) | FPGA.

    python scripts/board_compare.py --record 208 --sim                         # software FPGA model, live window
    python scripts/board_compare.py --record 208 --sim --seg annotation        # R positions from the annotations
    python scripts/board_compare.py --record 208 --sim --no-display            # headless: tables + files only
    python scripts/board_compare.py --record 208 --serial COM5 --speed 1       # real board (later)

Three label rows under the scrolling ECG, one letter per beat:
  MIT     the cardiologists' label from the PhysioNet annotation file (data/raw/mitdb/<rec>.atr, AAMI classes);
          with --seg detector an annotation the detector missed is drawn as an x, a detection without
          annotation as FP.
  float   the frozen Lite model before quantisation (model.pt, SHA-256 checked) on the SAME beat the FPGA
          receives: x = z_q / 2^4, RR features computed exactly from the same integer RR parts.  FPGA vs float
          therefore differ only by the int8 quantisation.
  FPGA    the label returned by the board (or the software FPGA model with --sim).
A letter is red where it differs from the MIT label.  The side panel keeps running confusion matrices and
macro-F1 of float and FPGA against MIT, the FPGA/float disagreements and the FPGA/int8-software check
(int_forward_q on the PC: must be bit-exact).

PC pre-processing (record_view: integer filter, Pan-Tompkins or annotated R, refinement, window, z-score, RR)
runs once on the whole record before the replay.  During the replay a beat is handed to the FPGA worker when
the playback reaches max(next R, R + delay + 180) -- the first sample at which its window and RR_next exist
(the real-time pipeline additionally waits for the detector to confirm the next R).  Its FPGA label appears
when the board answers.

Output (live mode on closing the window, or --no-display): results/<date>_<hash>_board_compare/ with
config.yaml, beats.csv, metrics.json, log.txt.  DS1 records are IN-SAMPLE for the final model (trained on
all DS1): the scores show agreement, not generalisation.
"""

import argparse
import csv
import hashlib
import json
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from python.data.constants import DS1  # noqa: E402
from python.data.mitbih import load_record  # noqa: E402
from python.experiment import Tee, make_results_dir  # noqa: E402
from python.models.metrics import format_report, report  # noqa: E402
from python.quant.pipeline_int import INPUT_EXP, quantise_rr_parts  # noqa: E402
from python.viz.common import CLASSES, FP, VF, load_frozen, record_view, rr_features_seconds  # noqa: E402
from scripts.board_eval import classify_with_retries  # noqa: E402
from scripts.board_monitor import open_board  # noqa: E402

COLORS = ("#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd")      # N, S, V, F
WRONG = "#d62728"
NAME = {0: "N", 1: "S", 2: "V", 3: "F", -1: "Q", FP: "FP", VF: "VF"}
NONE = -9                                                   # not answered (yet) / failed


def load_float_lite(fz):
    """The frozen Lite model before quantisation; model.pt must match export/final_models_manifest.json."""
    import torch

    from python.models.cnn1d import ECGCNN, ModelConfig
    man = json.loads((ROOT / "export" / "final_models_manifest.json").read_text(encoding="utf-8"))
    path = fz.lite_dir / "model.pt"
    if hashlib.sha256(path.read_bytes()).hexdigest() != man["lite"]["files"]["model.pt"]:
        raise RuntimeError(f"{path} does not match export/final_models_manifest.json (frozen model changed?)")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = ECGCNN(ModelConfig.from_dict(ck["model"]))
    m.load_state_dict(ck["state_dict"])
    return m.eval()


def float_predict(model, z_q: np.ndarray, rr_s: np.ndarray) -> np.ndarray:
    import torch
    if len(z_q) == 0:
        return np.zeros(0, dtype=np.int64)
    x = torch.as_tensor(z_q[:, None, :] / 2.0**INPUT_EXP, dtype=torch.float32)
    r = torch.as_tensor(rr_s, dtype=torch.float32)
    with torch.no_grad():
        return torch.cat([model(x[i:i + 1024], r[i:i + 1024]) for i in range(0, len(x), 1024)]).argmax(1).numpy()


class Comparison:
    """All beats of the window, the PC-side answers (float, int8 software) and the FPGA answers."""

    def __init__(self, rec, fz, seg: str, window: tuple[int, int]):
        self.fs, self.seg = fz.fs, seg
        v = record_view(rec, fz, seg)
        a, b = window
        bi = v["beat_index"]                                # R index of each z_q row
        rows = np.flatnonzero((v["r"][bi] >= a) & (v["r"][bi] < b))
        self.v, self.window = v, window
        self.ridx = bi[rows]                                # R index of each compared beat
        self.r = v["r"][self.ridx]
        self.label = v["label"][self.ridx]
        self.z_q = v["z_q"][rows]
        self.rr_q = quantise_rr_parts(v["rr_parts"][rows], fz.qm.rr_exp, fz.fs)
        self.sw_logits = v["logits"][self.ridx]
        self.sw_pred = v["pred"][self.ridx]
        self.float_pred = float_predict(load_float_lite(fz), self.z_q, rr_features_seconds(v["rr_parts"][rows], fz.fs))
        self.release = np.maximum(v["r"][self.ridx + 1], self.r + fz.delay + fz.half)   # beat_index never holds the last R
        n = len(self.r)
        self.fpga_pred = np.full(n, NONE, dtype=np.int64)
        self.fpga_logits = np.zeros((n, 4), dtype=np.int64)
        self.attempts = np.zeros(n, dtype=np.int64)
        self.link_s = np.full(n, np.nan)
        self.errors = [""] * n
        ann = v["ann_pos"]
        self.ann_sel = np.flatnonzero((ann >= a) & (ann < b))

    def set_fpga(self, k: int, res, attempts: int, err: str):
        self.attempts[k] = attempts
        if res is None:
            self.errors[k] = err
            return
        self.fpga_pred[k] = res.pred
        self.fpga_logits[k] = res.logits
        self.link_s[k] = res.seconds

    # ---- scores over the beats answered so far
    def scores(self, upto: int | None = None) -> dict:
        n = len(self.r) if upto is None else upto
        lab, fl, fp = self.label[:n], self.float_pred[:n], self.fpga_pred[:n]
        done = fp != NONE
        ok = done & (lab >= 0)
        agree = np.zeros((4, 4), dtype=np.int64)
        np.add.at(agree, (fp[done], fl[done]), 1)
        out = {"beats": int(n), "answered": int(done.sum()), "failed": int(sum(bool(e) for e in self.errors[:n])),
               "fpga_eq_int8_sw": int(np.sum(np.all(self.fpga_logits[:n][done] == self.sw_logits[:n][done], axis=1))),
               "fpga_ne_float": int(np.sum(fp[done] != fl[done])),
               "agreement_fpga_rows_float_cols": agree.tolist(),
               "float_vs_mit": report(lab[ok], fl[ok]) if ok.any() else None,
               "fpga_vs_mit": report(lab[ok], fp[ok]) if ok.any() else None}
        if self.seg == "detector":
            out["fp_detections"] = int(np.sum(lab == FP))
            out["in_vf"] = int(np.sum(lab == VF))
            if upto is None:
                missed = self.ann_sel[self.v["ann_missed"][self.ann_sel]]
                out["missed_annotations_NSVF"] = [int(np.sum(self.v["ann_label"][missed] == c)) for c in range(4)]
        return out

    def write(self, out: Path, log):
        with open(out / "beats.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["seg", "r", "t_s", "mit", "float", "fpga", "int8_sw", "fpga_eq_int8_sw", "attempts", "link_s",
                        "fpga_logit_N", "fpga_logit_S", "fpga_logit_V", "fpga_logit_F", "error"])
            for k in range(len(self.r)):
                p = int(self.fpga_pred[k])
                w.writerow([self.seg, int(self.r[k]), round(self.r[k] / self.fs, 3), NAME.get(int(self.label[k]), "?"),
                            CLASSES[int(self.float_pred[k])], CLASSES[p] if p != NONE else "",
                            CLASSES[int(self.sw_pred[k])],
                            int(p != NONE and np.array_equal(self.fpga_logits[k], self.sw_logits[k])),
                            int(self.attempts[k]), "" if np.isnan(self.link_s[k]) else round(float(self.link_s[k]), 4),
                            *(self.fpga_logits[k].tolist() if p != NONE else [""] * 4), self.errors[k]])
        m = self.scores()
        (out / "metrics.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
        log(f"\n{self.seg}: {m['beats']} beats, FPGA answered {m['answered']}, failed {m['failed']}, "
            f"FPGA == int8 software {m['fpga_eq_int8_sw']}/{m['answered']}, FPGA != float {m['fpga_ne_float']}")
        if m["float_vs_mit"]:
            log(f"\n--- Lite float vs MIT annotation ---\n{format_report(m['float_vs_mit'])}")
            log(f"\n--- FPGA vs MIT annotation ---\n{format_report(m['fpga_vs_mit'])}")
        log("\n--- FPGA (rows) vs float (cols), N/S/V/F ---")
        for row in m["agreement_fpga_rows_float_cols"]:
            log("  " + " ".join(f"{x:>6}" for x in row))
        if self.seg == "detector":
            log(f"detector: FP detections {m['fp_detections']}, missed annotations N/S/V/F "
                f"{m['missed_annotations_NSVF']}, in VF {m['in_vf']}")
        return m


class FpgaWorker:
    """Drives the board in a thread so the replay keeps real time."""

    def __init__(self, cmp: Comparison, clf, retries: int, pace: float):
        self.cmp, self.clf, self.retries, self.pace = cmp, clf, retries, pace
        self.jobs: queue.Queue = queue.Queue()
        self.pending = 0
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._run, daemon=True, name="fpga-worker")
        self.thread.start()

    def submit(self, k: int):
        with self.lock:
            self.pending += 1
        self.jobs.put(k)

    def _run(self):
        while True:
            k = self.jobs.get()
            if k is None:
                return
            t0 = time.perf_counter()
            res, attempts, err = classify_with_retries(self.clf, self.cmp.z_q[k], self.cmp.rr_q[k], self.retries)
            if self.pace > 0 and res is not None:
                time.sleep(max(0.0, res.seconds * self.pace - (time.perf_counter() - t0)))
            self.cmp.set_fpga(k, res, attempts, err)
            with self.lock:
                self.pending -= 1

    def stop(self):
        self.jobs.put(None)
        self.thread.join(timeout=5)


def _cm_text(rep) -> str:
    if rep is None:
        return "  -"
    lines = [f"  macro-F1 {rep['macro_f1']:.3f}  F1 " + " ".join(f"{c}{rep['per_class'][c]['f1']:.2f}" for c in CLASSES)]
    lines += ["        " + "".join(f"{c:>6}" for c in CLASSES)]
    lines += [f"  MIT {c} " + "".join(f"{x:>6}" for x in row) for c, row in zip(CLASSES, rep["confusion"])]
    return "\n".join(lines)


def run_live(cmp: Comparison, worker: FpgaWorker, speed: float, window_s: float, title: str):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    fs, v = cmp.fs, cmp.v
    sig = v["signal_mv"]
    a, b = cmp.window
    fig = plt.figure(figsize=(15, 7))
    fig.suptitle(title, fontsize=10)
    ax = fig.add_axes((0.10, 0.32, 0.56, 0.6))
    axl = fig.add_axes((0.10, 0.08, 0.56, 0.2), sharex=ax)
    axt = fig.add_axes((0.68, 0.03, 0.31, 0.92))
    axt.axis("off")
    (line,) = ax.plot([], [], lw=0.9, color="#333")
    ax.set_ylabel("mV (0.5-40 Hz integer filter)")
    axl.set_ylim(-0.5, 2.5)
    axl.set_yticks([2, 1, 0], ["MIT (PhysioNet)", "Lite float", "FPGA"])
    axl.set_xlabel("time (s)")
    txt = axt.text(0, 1, "", va="top", family="monospace", fontsize=8)
    artists = []
    state = {"t0": time.perf_counter(), "next": 0}
    ann_pos, ann_lab, ann_missed = v["ann_pos"], v["ann_label"], v["ann_missed"]

    def update(_):
        t = a / fs + (time.perf_counter() - state["t0"]) * speed
        s = min(int(t * fs), b)
        while state["next"] < len(cmp.r) and (cmp.release[state["next"]] <= s or s >= b):   # window end: release the rest
            worker.submit(state["next"])
            state["next"] += 1
        lo_s = max(a, s - int(window_s * fs))
        tt = np.arange(lo_s, s) / fs
        y = sig[lo_s:s]
        line.set_data(tt, y)
        x0 = max(a / fs, t - window_s)
        ax.set_xlim(x0, x0 + window_s)
        if len(y):
            lo, hi = float(np.min(y)), float(np.max(y))
            ax.set_ylim(lo - 0.15 * (hi - lo + 1e-3), hi + 0.15 * (hi - lo + 1e-3))
        for art in artists:
            art.remove()
        artists.clear()
        # MIT row: every annotation in view (Q grey, missed by the detector = x)
        for j in np.flatnonzero((ann_pos >= lo_s) & (ann_pos < s)):
            c = int(ann_lab[j])
            col = COLORS[c] if c >= 0 else "#7f7f7f"
            artists.append(axl.text(ann_pos[j] / fs, 2, NAME.get(c, "?"), ha="center", va="center", color=col,
                                    fontsize=11, weight="bold"))
            if ann_missed[j]:
                artists.append(axl.text(ann_pos[j] / fs, 1.5, "x missed", ha="center", va="center", color=WRONG,
                                        fontsize=7))
        # float and FPGA rows at the analysed R positions
        for k in np.flatnonzero((cmp.r >= lo_s) & (cmp.r < s)):
            lab, x = int(cmp.label[k]), cmp.r[k] / fs
            ax_y = sig[cmp.r[k]]
            artists.append(ax.plot([x], [ax_y], "o", ms=5, color="#555")[0])
            if lab == FP:
                artists.append(axl.text(x, 2, "FP", ha="center", va="center", color=WRONG, fontsize=8))
            released = k < state["next"]
            for row, p in ((1, int(cmp.float_pred[k]) if released else NONE), (0, int(cmp.fpga_pred[k]))):
                if p == NONE:
                    if row == 0 and released:
                        artists.append(axl.text(x, 0, "…" if not cmp.errors[k] else "ERR", ha="center",
                                                va="center", color="#7f7f7f", fontsize=9))
                    continue
                wrong = lab >= 0 and p != lab
                artists.append(axl.text(x, row, CLASSES[p], ha="center", va="center", fontsize=11, weight="bold",
                                        color=WRONG if wrong else COLORS[p]))
        n = state["next"]
        m = cmp.scores(n)
        mean_link = np.nanmean(cmp.link_s[:n]) if np.any(~np.isnan(cmp.link_s[:n])) else float("nan")
        hw = worker.clf.hw
        txt.set_text(
            f"record replay t = {t:7.1f} s  (x{speed:g})\n"
            f"segmentation: {cmp.seg}\n\n"
            f"beats released {n}/{len(cmp.r)}, FPGA answered {m['answered']}\n"
            f"FPGA queue {worker.pending}, failed {m['failed']}\n"
            f"FPGA link {mean_link:.2f} s/beat, {hw.rows}x{hw.cols} PE\n"
            f"FPGA == int8 software: {m['fpga_eq_int8_sw']}/{m['answered']}\n"
            f"FPGA != Lite float:    {m['fpga_ne_float']}\n"
            + (f"FP detections {m['fp_detections']}\n" if cmp.seg == "detector" else "")
            + f"\nLite float vs MIT (answered beats)\n{_cm_text(m['float_vs_mit'])}\n"
            f"\nFPGA vs MIT\n{_cm_text(m['fpga_vs_mit'])}\n"
            f"\nred letter = differs from MIT label")
        if s >= b:
            ax.set_title("end of the replay window" + ("" if worker.pending == 0 else f", waiting for {worker.pending} FPGA answers"))

    anim = FuncAnimation(fig, update, interval=50, cache_frame_data=False)
    plt.show()
    return anim


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--record", type=int, required=True)
    link = ap.add_mutually_exclusive_group(required=True)
    link.add_argument("--serial", help="serial port of the board, e.g. COM5 (needs pyserial)")
    link.add_argument("--sim", action="store_true", help="software FPGA model instead of a board")
    ap.add_argument("--seg", choices=("detector", "annotation"), default="detector",
                    help="R positions: integer Pan-Tompkins (deployment) or the annotated R")
    ap.add_argument("--start", type=float, default=0.0, help="window start (s)")
    ap.add_argument("--dur", type=float, default=None, help="window length (s), default to the end of the record")
    ap.add_argument("--speed", type=float, default=1.0, help="replay speed, 1 = real time")
    ap.add_argument("--window", type=float, default=10.0, help="seconds of ECG on screen")
    ap.add_argument("--sim-pace", type=float, default=1.0,
                    help="--sim: sleep the simulated link time x this factor per beat (1 = behaves like the real link)")
    ap.add_argument("--baud", type=int, default=1500000, help="must match the bitstream")
    ap.add_argument("--timeout", type=float, default=3.0, help="per-command response timeout (s)")
    ap.add_argument("--retries", type=int, default=1, help="resend a failed beat from scratch this many times")
    ap.add_argument("--no-display", action="store_true", help="no window: classify every beat, print tables, write files")
    args = ap.parse_args(argv)

    fz = load_frozen()
    rec = load_record(args.record)                       # DS2 -> error unless the user set ECG_ALLOW_DS2=1
    a = int(round(args.start * fz.fs))
    b = len(rec.adc) if args.dur is None else min(len(rec.adc), int(round((args.start + args.dur) * fz.fs)))
    if b <= a:
        raise SystemExit("empty window")
    cfg = {"name": "board_compare", "record": args.record, "seg": args.seg, "start_s": args.start,
           "dur_s": (b - a) / fz.fs, "link": "sim" if args.sim else args.serial, "baud": args.baud,
           "retries": args.retries, "lite": fz.lite_dir.name, "mode": "headless" if args.no_display else "live"}
    out = make_results_dir(cfg, "_board_compare")
    log = Tee(out / "log.txt")
    note = "DS1: IN-SAMPLE (final model trained on all DS1)" if args.record in DS1 else "not DS1"
    log(f"record {args.record}, samples {a}..{b} ({(b - a) / fz.fs:.0f} s), segmentation {args.seg} -- {note}")
    log("PC pre-processing of the whole record ...")
    cmp = Comparison(rec, fz, args.seg, (a, b))
    log(f"{len(cmp.r)} beats to classify; Lite float done on the PC")

    transport, clf = open_board(args)
    hw = clf.hw
    log(f"FPGA INFO: {hw.rows}x{hw.cols} PE, {hw.clk_hz / 1e6:g} MHz, {hw.baud} baud, features 0x{hw.features:02X}"
        + ("" if hw.baud == args.baud else f"  WARNING: port opened at {args.baud}"))
    try:
        if args.no_display:
            t0 = time.perf_counter()
            for k in range(len(cmp.r)):
                cmp.set_fpga(k, *classify_with_retries(clf, cmp.z_q[k], cmp.rr_q[k], args.retries))
                if (k + 1) % 200 == 0 or k + 1 == len(cmp.r):
                    log(f"  {k + 1}/{len(cmp.r)} beats on the FPGA, {time.perf_counter() - t0:.0f} s")
        else:
            worker = FpgaWorker(cmp, clf, args.retries, args.sim_pace if args.sim else 0.0)
            title = (f"Record {args.record} (PhysioNet MIT-BIH) -- MIT annotation | Lite float | "
                     f"{'FPGA (software model)' if args.sim else 'FPGA ' + args.serial} -- {note}")
            run_live(cmp, worker, args.speed, args.window, title)
            worker.stop()
    finally:
        transport.close()
        m = cmp.write(out, log)
        log(f"results: {out}")
        log.close()
    ok = m["failed"] == 0 and m["fpga_eq_int8_sw"] == m["answered"]
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
