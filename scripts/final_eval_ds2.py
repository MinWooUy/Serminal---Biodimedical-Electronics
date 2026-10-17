"""FINAL evaluation on DS2 (MIT-BIH inter-patient test set) -- RUN ONCE, BY THE USER ONLY.

    PowerShell:  $env:ECG_ALLOW_DS2 = "1"
                 .venv\\Scripts\\python.exe scripts\\final_eval_ds2.py --final-lite results\\<..._final_A5_svdb_n25_lite_e3> `
                                                                     --final-full results\\<..._final_cnn_full>

Claude must never set ECG_ALLOW_DS2, run this script, or tune anything on its output (CLAUDE.md rule 1).
All models and parameters are frozen (NHATKY 2026-10-15); this script only measures and the results are
reported as they are. It refuses to run on a dirty working tree and records `git rev-parse HEAD`, and it
refuses to run unless the final model files match export/final_models_manifest.json (SHA-256) and
export/weights.h regenerated from <final-lite>/lite_int8.npz is byte-identical to the committed one
(verify_provenance); all hashes are written to the results.

Models: Full float, Lite float, Lite fully integer (integer filter / refinement / z-score / RR + int8 CNN,
the golden reference of the C port).
Evaluations on the 22 DS2 records:
  1. Detector: Se / +P at +-150 ms (and +-75 ms), per record and total, VF segments excluded.
  2. Classifier with annotation-based segmentation (R = annotations re-aligned with the abs rule; integer
     path: integer refinement on the integer-filtered signal); Q beats dropped after RR.
  3. End-to-end with the detector (R = detector, as deployed), labels by one-to-one matching (+-150 ms):
     3a. matched beats only;
     3b. gross: a missed beat (detector FN) counts against Se of its true class; an unmatched detection
         (FP) counts as a false positive of the class it is predicted as -- for the 4-class macro-F1 this
         includes N; SVEB / VEB Se and +P follow EC57 (FP predicted S/V counted; true-F beats predicted V
         are not VEB false positives).
     Detections matched to a Q annotation and detections / annotations inside VF segments are removed
     from every count (python: e2e_counts).
Every evaluation reports macro-F1 (4 classes), macro-F1 N/S/V, per-class P/R/F1, SVEB/VEB Se/+P and the
confusion matrix. Output: results/FINAL_DS2_<date>_<hash>/ (config.yaml, metrics.json, log.txt,
detector_per_record.csv) with a summary table at the end of the log.
"""

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from python.data.constants import require_ds2_access  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
CLASS_NAMES = ("N", "S", "V", "F")
NO_PRED = -9            # detection that was not classified (first / last beat of a record, incomplete window)


def main() -> None:
    # DS2 lock FIRST: nothing below runs (no file is opened) unless the user set ECG_ALLOW_DS2=1.
    require_ds2_access()
    import argparse as _ap
    from python.data.splits import get_ds2_records
    status = subprocess.run(["git", "status", "--porcelain"], cwd=REPO, capture_output=True, text=True, check=True)
    if status.stdout.strip():
        print("Working tree is not clean (git status --porcelain):\n" + status.stdout +
              "\nCommit or stash the changes first. DS2 was NOT evaluated.")
        sys.exit(2)
    git_rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True,
                             check=True).stdout.strip()
    ap = _ap.ArgumentParser()
    ap.add_argument("--final-lite", required=True)
    ap.add_argument("--final-full", required=True)
    args = ap.parse_args()
    provenance = verify_provenance(Path(args.final_lite), Path(args.final_full))     # exits on any mismatch
    run(get_ds2_records(), Path(args.final_lite), Path(args.final_full), prefix="FINAL_DS2", git_rev=git_rev,
        provenance=provenance)


def verify_provenance(fl: Path, ff: Path, manifest_path: Path = REPO / "export" / "final_models_manifest.json") -> dict:
    """Refuse to evaluate unless the final models are exactly the frozen ones:
    1. every file of both final directories has the SHA-256 recorded in export/final_models_manifest.json
       (same directory names, same file set);
    2. export/weights.h has the recorded SHA-256;
    3. weights.h REGENERATED from <final-lite>/lite_int8.npz is byte-identical to export/weights.h, i.e. the
       'Lite integer' branch (which loads that same lite_int8.npz and never re-quantises) uses exactly the
       constants of weights.h.
    Returns the hashes (stored in the results). Exits with code 3 on any mismatch."""
    import hashlib

    def sha(b: bytes) -> str:
        return hashlib.sha256(b).hexdigest()

    def fail(msg: str):
        print(f"PROVENANCE CHECK FAILED: {msg}\nDS2 was NOT evaluated.")
        sys.exit(3)

    man = json.loads(manifest_path.read_text(encoding="utf-8"))
    hashes = {}
    for key, d in (("lite", fl), ("full", ff)):
        if d.name != man[key]["dir"]:
            fail(f"{key} directory {d.name} != manifest {man[key]['dir']}")
        now = {p.relative_to(d).as_posix(): sha(p.read_bytes()) for p in sorted(d.rglob("*")) if p.is_file()}
        if now != man[key]["files"]:
            diff = sorted(set(now.items()) ^ set(man[key]["files"].items()))
            fail(f"{key} files differ from the manifest: {diff}")
        hashes[key] = now
    wh = (REPO / "export" / "weights.h").read_bytes()
    if sha(wh) != man["weights_h_sha256"]:
        fail("export/weights.h does not match the manifest")
    from python.dsp.filters import group_delay_compensation
    from python.dsp.pan_tompkins import detector_from_config
    from python.experiment import load_config
    from python.quant.int8 import load_quant_model
    from scripts.export_c import weights_h_text
    bcfg = load_config(REPO / load_config(fl / "config.yaml")["beats_config"])
    pt, sos = detector_from_config(load_config(REPO / bcfg["detector_config"]))
    regen = weights_h_text(load_quant_model(fl / "lite_int8.npz"), pt, sos, group_delay_compensation(sos, pt.fs),
                           bcfg["beats"]).encode("ascii")
    if regen != wh:
        fail("weights.h regenerated from lite_int8.npz differs from export/weights.h")
    hashes["weights_h_sha256"] = sha(wh)
    hashes["weights_h_regenerated_from"] = f"{fl.name}/lite_int8.npz"
    print(f"provenance OK: {len(hashes['lite'])} Lite files, {len(hashes['full'])} Full files, weights.h "
          f"{hashes['weights_h_sha256'][:16]}... regenerated identically from lite_int8.npz")
    return hashes


def e2e_counts(ann_pos: np.ndarray, ann_aami: np.ndarray, vf: np.ndarray, det_pos: np.ndarray,
               det_pred: np.ndarray, tol: int) -> dict:
    """End-to-end bookkeeping for one record.

    Annotations and detections inside VF segments are removed; the rest are matched one-to-one (+-tol).
    Returns y/p of matched beats (Q-matched pairs and unclassified detections removed), FN per class
    (unmatched non-Q annotations) and the predictions of FP detections (unmatched, classified).
    """
    from python.dsp.evaluate_detector import match_beats, outside_vf
    ann_pos, ann_aami = np.asarray(ann_pos), np.asarray(ann_aami)
    det_pos, det_pred = np.asarray(det_pos), np.asarray(det_pred)
    ka, kd = outside_vf(ann_pos, vf), outside_vf(det_pos, vf)
    ai, di = np.flatnonzero(ka), np.flatnonzero(kd)
    m = match_beats(ann_pos[ai], det_pos[di], tol)
    y = ann_aami[ai[m.pairs[:, 0]]].astype(np.int64)
    p = det_pred[di[m.pairs[:, 1]]].astype(np.int64)
    keep = (y >= 0) & (p != NO_PRED)                       # Q-matched detections and unclassified ones drop out
    missed = ann_aami[ai[m.fn]]
    fp = det_pred[di[m.fp]]
    return {"y": y[keep], "p": p[keep], "fn": np.array([int(np.sum(missed == c)) for c in range(4)]),
            "fp_pred": fp[fp != NO_PRED].astype(np.int64)}


def gross_metrics(y: np.ndarray, p: np.ndarray, fp_pred: np.ndarray, fn: np.ndarray) -> dict:
    """3b: per-class Se includes missed beats; per-class +P includes FP detections predicted as that class
    (also N) -> 4-class macro-F1. SVEB / VEB follow EC57 (VEB +P excludes true-F predicted V)."""
    cm = np.zeros((4, 4), dtype=np.int64)
    np.add.at(cm, (np.asarray(y, dtype=np.int64), np.asarray(p, dtype=np.int64)), 1)
    fp_c = np.bincount(np.asarray(fp_pred, dtype=np.int64), minlength=4)
    fn = np.asarray(fn)
    per, f1s = {}, []
    for c, name in enumerate(CLASS_NAMES):
        tp = cm[c, c]
        se = tp / max(cm[c].sum() + fn[c], 1)
        ppv = tp / max(cm[:, c].sum() + fp_c[c], 1)
        f1 = 2 * se * ppv / (se + ppv) if se + ppv > 0 else 0.0
        per[name] = {"se": float(se), "ppv": float(ppv), "f1": float(f1), "missed": int(fn[c]),
                     "fp_detections_predicted": int(fp_c[c])}
        f1s.append(f1)
    S, V, F = 1, 2, 3
    sveb = {"se": float(cm[S, S] / max(cm[S].sum() + fn[S], 1)),
            "ppv": float(cm[S, S] / max(cm[:, S].sum() + fp_c[S], 1))}
    veb = {"se": float(cm[V, V] / max(cm[V].sum() + fn[V], 1)),
           "ppv": float(cm[V, V] / max(cm[:, V].sum() - cm[F, V] + fp_c[V], 1))}
    return {"macro_f1": float(np.mean(f1s)), "macro_f1_nsv": float(np.mean(f1s[:3])), "per_class": per,
            "ec57": {"SVEB": sveb, "VEB": veb}, "confusion": cm.tolist(),
            "missed_per_class": fn.tolist(), "fp_detections_predicted": fp_c.tolist()}


def run(records, fl: Path, ff: Path, prefix: str, git_rev: str = "", provenance: dict | None = None) -> dict:
    """The evaluation itself (no DS2 lock here: main() checks it first, and load_record locks DS2 again).
    tests/ call it on ONE DS1 record as a smoke test only -- never on DS2."""
    import datetime as dt

    import yaml

    from python.data.beats import BeatConfig, build_record
    from python.data.mitbih import REPO_ROOT, load_record
    from python.dsp.evaluate_detector import evaluate_record, se_ppv, tol_samples
    from python.dsp.pan_tompkins import detect_offline, detector_from_config
    from python.dsp.realign import half_window_samples, realign
    from python.experiment import Tee, config_hash, load_config
    from python.models.cnn1d import ECGCNN, ModelConfig
    from python.models.metrics import format_report, report
    from python.quant.int8 import int_forward_q, load_quant_model
    from python.quant.pipeline_int import filter_int, run_integer_beats
    from scripts.pipeline_int_eval import quantise_rr_parts

    lite_cfg = load_config(fl / "config.yaml")
    bcfg = load_config(REPO_ROOT / lite_cfg["beats_config"])
    det_cfg = load_config(REPO_ROOT / bcfg["detector_config"])
    pt, sos = detector_from_config(det_cfg)
    half = int(bcfg["beats"]["half_window"])
    lo, hi = (int(round(v * pt.fs)) for v in bcfg["beats"]["rr_clip_s"])
    tol150 = tol_samples(150, pt.fs)

    def load_float(d: Path) -> ECGCNN:
        ck = torch.load(d / "model.pt", weights_only=False)
        m = ECGCNN(ModelConfig.from_dict(ck["model"]))
        m.load_state_dict(ck["state_dict"])
        return m.eval()

    full, lite = load_float(ff), load_float(fl)
    qm = load_quant_model(fl / "lite_int8.npz")      # the SAME parameters that generated export/weights.h
    run_cfg = {"name": prefix, "lite": fl.name, "full": ff.name, "beats": bcfg["beats"], "records": list(records),
               "git_rev": git_rev, "provenance": provenance or {}}
    out = REPO_ROOT / "results" / f"{prefix}_{dt.date.today().isoformat()}_{config_hash(run_cfg)}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(run_cfg, sort_keys=False), encoding="utf-8")
    log = Tee(out / "log.txt")
    log(f"evaluation on records {list(records)}; git HEAD {git_rev or '(not recorded)'}")
    log(f"frozen models: Full {ff.name}, Lite {fl.name} (+ lite_int8.npz)")
    if provenance:
        log(f"provenance: weights.h SHA-256 {provenance['weights_h_sha256']} (regenerated from "
            f"{provenance['weights_h_regenerated_from']}); model file hashes in config.yaml / metrics.json")

    @torch.no_grad()
    def predict(m: ECGCNN, X: np.ndarray, rr: np.ndarray) -> np.ndarray:
        if len(X) == 0:
            return np.zeros(0, dtype=np.int64)
        x = torch.as_tensor(X[:, None, :], dtype=torch.float32)
        r = torch.as_tensor(rr, dtype=torch.float32)
        return torch.cat([m(x[i:i + 2048], r[i:i + 2048]) for i in range(0, len(x), 2048)]).argmax(1).numpy()

    models = ("full_float", "lite_float", "lite_integer")
    ann = {m: {"y": [], "p": []} for m in models}
    e2e = {m: {"y": [], "p": [], "fp_pred": [], "fn": np.zeros(4, dtype=np.int64)} for m in models}
    det_rows, det_tot = [], {"150ms": np.zeros(3, np.int64), "75ms": np.zeros(3, np.int64)}

    for rid in records:
        rec = load_record(rid)                         # second DS2 lock inside the loader
        # ---- 1. detector
        res = detect_offline(rec.adc, rec.adc_zero, rec.adc_gain, pt, sos)
        ann_re, _ = realign(rec.r_samples, res.x_filt, res.delay, half_window_samples(pt.fs), pt.realign_mode)
        ev = evaluate_record(rec.r_samples, res.r_refined, res.r_coarse, ann_re, rec.vf_segments, pt.fs)
        row = {"rec": rid}
        for k in ("150ms", "75ms"):
            det_tot[k] += [ev[k]["tp"], ev[k]["fn"], ev[k]["fp"]]
            row |= {f"tp_{k}": ev[k]["tp"], f"fn_{k}": ev[k]["fn"], f"fp_{k}": ev[k]["fp"],
                    f"se_{k}": ev[k]["se"], f"ppv_{k}": ev[k]["ppv"]}
        det_rows.append(row)
        y_int = filter_int(rec.adc, rec.adc_zero, sos)

        # ---- 2. annotation-based segmentation (Q dropped after RR)
        a_arr, _ = build_record(rec, pt, sos, BeatConfig.from_dict({**bcfg["beats"], "rr_source": "annotation"}))
        sel = a_arr["y"] >= 0
        for name, m in (("full_float", full), ("lite_float", lite)):
            ann[name]["y"].append(a_arr["y"][sel])
            ann[name]["p"].append(predict(m, a_arr["X"][sel], a_arr["rr"][sel]))
        ib = run_integer_beats(y_int, rec.r_samples, res.delay, half, lo, hi)
        lab = rec.aami[ib["order"]][ib["beat_index"]]
        keep = lab >= 0
        ann["lite_integer"]["y"].append(lab[keep])
        ann["lite_integer"]["p"].append(
            int_forward_q(qm, ib["z_q"][keep], quantise_rr_parts(ib["rr_parts"][keep], qm.rr_exp)).argmax(1))

        # ---- 3. end-to-end with the detector (same bookkeeping for every model: e2e_counts)
        d_arr, _ = build_record(rec, pt, sos, BeatConfig.from_dict({**bcfg["beats"], "rr_source": "detector"}))
        det_sorted = np.sort(res.r_refined)
        pos = np.searchsorted(det_sorted, d_arr["r_pos"])
        assert np.array_equal(det_sorted[pos], d_arr["r_pos"])
        for name, m in (("full_float", full), ("lite_float", lite)):
            det_pred = np.full(len(det_sorted), NO_PRED, dtype=np.int64)
            det_pred[pos] = predict(m, d_arr["X"], d_arr["rr"])
            c = e2e_counts(rec.r_samples, rec.aami, rec.vf_segments, det_sorted, det_pred, tol150)
            for k in ("y", "p", "fp_pred"):
                e2e[name][k].append(c[k])
            e2e[name]["fn"] += c["fn"]
        ib = run_integer_beats(y_int, res.r_coarse, res.delay, half, lo, hi)
        det_pred = np.full(len(ib["r_int"]), NO_PRED, dtype=np.int64)
        det_pred[ib["beat_index"]] = int_forward_q(qm, ib["z_q"], quantise_rr_parts(ib["rr_parts"], qm.rr_exp)).argmax(1)
        c = e2e_counts(rec.r_samples, rec.aami, rec.vf_segments, ib["r_int"], det_pred, tol150)
        for k in ("y", "p", "fp_pred"):
            e2e["lite_integer"][k].append(c[k])
        e2e["lite_integer"]["fn"] += c["fn"]
        log(f"  {rid}: detector Se {100 * ev['150ms']['se']:.2f} +P {100 * ev['150ms']['ppv']:.2f}; "
            f"annotation beats {int(sel.sum())}")

    tp, fn, fp = (int(v) for v in det_tot["150ms"])
    se, ppv = se_ppv(tp, fn, fp)
    tp75, fn75, fp75 = (int(v) for v in det_tot["75ms"])
    se75, ppv75 = se_ppv(tp75, fn75, fp75)
    results = {"git_rev": git_rev, "records": list(records), "provenance": provenance or {},
               "detector": {"150ms": {"tp": tp, "fn": fn, "fp": fp, "se": se, "ppv": ppv},
                            "75ms": {"tp": tp75, "fn": fn75, "fp": fp75, "se": se75, "ppv": ppv75}},
               "annotation": {}, "end_to_end_matched": {}, "end_to_end_gross": {}}
    for name in models:
        r = report(np.concatenate(ann[name]["y"]), np.concatenate(ann[name]["p"]))
        results["annotation"][name] = r
        log(f"\n=== 2. CLASSIFIER, annotation segmentation — {name} ===\n{format_report(r)}")
    for name in models:
        y, p = np.concatenate(e2e[name]["y"]), np.concatenate(e2e[name]["p"])
        g = gross_metrics(y, p, np.concatenate(e2e[name]["fp_pred"]), e2e[name]["fn"])
        results["end_to_end_matched"][name] = report(y, p)
        results["end_to_end_gross"][name] = g
        log(f"\n=== 3a. END-TO-END, matched beats — {name} ===\n{format_report(results['end_to_end_matched'][name])}")
        log(f"=== 3b. END-TO-END gross — {name}: missed per class {g['missed_per_class']}, "
            f"FP detections predicted per class {g['fp_detections_predicted']}")

    # ---- summary table
    log("\n" + "=" * 110 + "\nSUMMARY")
    log(f"Detector ({len(records)} records): +-150 ms Se {100 * se:.3f}%  +P {100 * ppv:.3f}%  "
        f"(TP {tp} FN {fn} FP {fp});  +-75 ms Se {100 * se75:.3f}%  +P {100 * ppv75:.3f}%")
    log(f"{'model':<13} {'mode':<20} {'macro-F1':>8} {'N/S/V':>7} {'SVEB Se':>8} {'SVEB +P':>8} {'VEB Se':>8} {'VEB +P':>8}")
    modes = (("annotation", "annotation"), ("e2e matched", "end_to_end_matched"), ("e2e gross", "end_to_end_gross"))
    for name in models:
        for label, key in modes:
            r = results[key][name]
            e = r["ec57"]
            log(f"{name:<13} {label:<20} {r['macro_f1']:8.4f} {r['macro_f1_nsv']:7.4f} {e['SVEB']['se']:8.4f} "
                f"{e['SVEB']['ppv']:8.4f} {e['VEB']['se']:8.4f} {e['VEB']['ppv']:8.4f}")
    log("\nconfusion matrices (rows true N/S/V/F, cols predicted N/S/V/F; gross adds missed beats and FP detections):")
    for name in models:
        for label, key in modes:
            cm = results[key][name]["confusion"]
            extra = ""
            if key == "end_to_end_gross":
                g = results[key][name]
                extra = f"   missed {g['missed_per_class']}   FP detections predicted {g['fp_detections_predicted']}"
            log(f"  {name:<13} {label:<12} " + " | ".join(" ".join(f"{v:>6}" for v in row) for row in cm) + extra)
    with open(out / "detector_per_record.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(det_rows[0]))
        w.writeheader()
        w.writerows(det_rows)
    (out / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    log(f"\nsaved -> {out}")
    log.close()
    return results


if __name__ == "__main__":
    main()
