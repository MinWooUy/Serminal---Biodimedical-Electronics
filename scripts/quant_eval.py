"""Quantisation gate: int8 integer inference vs float, pooled OOF macro-F1 (DS1, record-grouped CV).

For every (seed, fold) checkpoint of a finished CV run: fold BN / 1/45 / RR mu-sigma, calibrate the
power-of-2 activation exponents on the fold's INNER-TRAIN beats (never the outer fold), run the numpy
integer model (python/quant/int8.py, golden reference for C) on the outer fold, and pool the OOF
predictions. Sanity check: the float folded model reproduces the saved float OOF predictions.
Gate: macro-F1 drop (float - int8, mean over seeds) <= 0.02.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from python.experiment import Tee, load_config  # noqa: E402
from python.models.cnn1d import ECGCNN, ModelConfig  # noqa: E402
from python.models.cv_data import attach_extra_train, load_cv_data, with_extra_train  # noqa: E402
from python.models.metrics import format_report, mean_std, report  # noqa: E402
from python.models.training import TrainConfig, fold_indices  # noqa: E402
from python.quant.int8 import float_forward, fold_model, int_forward, quantize_model  # noqa: E402

MAX_DROP = 0.02


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="CV result dir with ckpt/ and oof files")
    ap.add_argument("--oof", default="oof_s{seed}.npz", help="float OOF file pattern to check against")
    ap.add_argument("--calib", default="max", help="max | p99.99 | ...")
    ap.add_argument("--n-calib", type=int, default=4000, help="inner-train beats used for calibration")
    ap.add_argument("--ckpt", default="s{seed}_f{k}.pt", help="checkpoint name pattern in ckpt/")
    args = ap.parse_args()
    run = Path(args.run)
    cfg = load_config(run / "config.yaml")
    cv = load_cv_data(cfg["beats_config"])
    # calibration uses the (combined) inner-train, i.e. DS1 + training-only extra beats if the run had them
    d, n_main, extra_idx, extra_name = attach_extra_train(cfg, cv.data)
    y_main = d["y"][:n_main]
    tcfg = TrainConfig.from_dict(cfg["train"])
    log = Tee(run / f"quant_{args.calib}.log")
    log(f"run {run.name}, calibration {args.calib} on {args.n_calib} inner-train beats per fold"
        + (f" (inner-train includes {extra_name})" if extra_name else ""))
    rep_f, rep_q, agree, max_dev, transitions, flip_stats = [], [], [], 0.0, [], []
    for seed in cfg["seeds"]:
        saved = np.load(run / args.oof.format(seed=seed))["probs"]
        pred_f = np.full(n_main, -1)
        pred_q = np.full(n_main, -1)
        logit_f = np.zeros((n_main, 4))
        logit_q = np.zeros((n_main, 4))            # int8-path logits, dequantised to logit units
        for k in cfg["folds"]:
            ck = torch.load(run / "ckpt" / args.ckpt.format(seed=seed, k=k), weights_only=False)
            m = ECGCNN(ModelConfig.from_dict(ck["model"]))
            m.load_state_dict(ck["state_dict"])
            idx = fold_indices(d["rec_id"][:n_main], d["r_pos"][:n_main], cv.rec_len, cv.folds, k, cv.delay,
                               cv.half, tcfg, cv.fs)
            idx = with_extra_train(idx, extra_idx, n_main)
            rng = np.random.default_rng(seed * 10 + k)
            cal = np.sort(rng.choice(idx["train"], size=min(args.n_calib, len(idx["train"])), replace=False))
            fm = fold_model(m)
            qm = quantize_model(fm, d["X"][cal], d["rr"][cal], args.calib)
            out = idx["outer"]
            lf = float_forward(fm, d["X"][out], d["rr"][out])
            pf = np.exp(lf - lf.max(1, keepdims=True))
            pf /= pf.sum(1, keepdims=True)
            max_dev = max(max_dev, float(np.abs(pf - saved[out]).max()))
            pred_f[out] = lf.argmax(1)
            li = int_forward(qm, d["X"][out], d["rr"][out])
            pred_q[out] = li.argmax(1)
            logit_f[out] = lf
            logit_q[out] = li * 2.0 ** -(qm.fc2["k"] + qm.fc2["a_in"])
            exps = [(c["a_out"], int(c["k"].min()), int(c["k"].max())) for c in qm.convs]
            log(f"[seed {seed} fold {k}] act exps / weight exps per conv {exps}, rr_exp {qm.rr_exp.tolist()}, "
                f"fc1 a_out {qm.fc1['a_out']}; argmax agreement int8 vs float "
                f"{np.mean(pred_q[out] == pred_f[out]):.4f}")
        assert np.all(pred_f >= 0) and np.all(pred_q >= 0)
        np.savez_compressed(run / f"quant_{args.calib}_logits_s{seed}.npz", logit_float=logit_f, logit_int8=logit_q,
                            y=y_main, fold=np.load(run / args.oof.format(seed=seed))["fold"])
        rep_f.append(report(y_main, pred_f))
        rep_q.append(report(y_main, pred_q))
        agree.append(float(np.mean(pred_q == pred_f)))
        # label transition float -> int8 (rows = float argmax N/S/V/F, cols = int8 argmax); report only
        tr = np.zeros((4, 4), dtype=np.int64)
        np.add.at(tr, (pred_f, pred_q), 1)
        transitions.append(tr.tolist())
        flipped = pred_f != pred_q
        flips_true = {c: int(np.sum(flipped & (y_main == i))) for i, c in enumerate(("N", "S", "V", "F"))}
        flip_stats.append(flips_true)
        log(f"seed {seed}: beats whose label changed float -> int8, by TRUE class: {flips_true} "
            f"(total {int(flipped.sum())})")
        log(f"seed {seed}: same argmax float/int8 {agree[-1]:.4f}; transition float -> int8 (rows float N/S/V/F):\n"
            + "\n".join("  " + " ".join(f"{v:>6}" for v in row) for row in tr))
        log(f"\n=== seed {seed} float (folded numpy) ===\n{format_report(rep_f[-1])}")
        log(f"=== seed {seed} int8 ===\n{format_report(rep_q[-1])}")
    drops = [f["macro_f1"] - q["macro_f1"] for f, q in zip(rep_f, rep_q)]
    msf, msq = mean_std(rep_f), mean_std(rep_q)
    log(f"\nsanity: max |softmax(folded float) - saved OOF prob| = {max_dev:.2e}")
    log(f"argmax agreement int8 vs float per seed: {[round(a, 4) for a in agree]}")
    log(f"macro-F1 float {msf['macro_f1']['mean']:.4f} ± {msf['macro_f1']['std']:.4f} | "
        f"int8 {msq['macro_f1']['mean']:.4f} ± {msq['macro_f1']['std']:.4f} | drop per seed "
        f"{[round(x, 4) for x in drops]} mean {np.mean(drops):+.4f}")
    for g in ("SVEB", "VEB"):
        log(f"  {g} Se/+P float {msf['ec57'][g]['se']['mean']:.3f}/{msf['ec57'][g]['ppv']['mean']:.3f} "
            f"int8 {msq['ec57'][g]['se']['mean']:.3f}/{msq['ec57'][g]['ppv']['mean']:.3f}")
    ok = np.mean(drops) <= MAX_DROP
    log(f"QUANT GATE (mean drop <= {MAX_DROP}): {'PASS' if ok else 'FAIL'}")
    (run / f"quant_{args.calib}.json").write_text(json.dumps(
        {"calib": args.calib, "float": rep_f, "int8": rep_q, "drops": drops, "agreement": agree,
         "transition_float_to_int8": transitions, "flips_by_true_class": flip_stats,
         "max_prob_dev": max_dev, "pass": bool(ok)}, indent=2), encoding="utf-8")
    log.close()


if __name__ == "__main__":
    main()
