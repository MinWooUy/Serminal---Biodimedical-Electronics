"""Record-grouped 5-fold CV for the CNN (DS1 only).

Per (seed, outer fold k): inner-train / inner-val split in time inside the
train records (see python/models/training.py); early stopping and LR plateau on
the weighted inner-val loss; the outer fold only receives out-of-fold predictions.
When all folds of a seed are done: macro-F1 on the pooled OOF predictions.

  python scripts/train_cv.py --config configs/cnn_full.yaml               # all folds x seeds
  python scripts/train_cv.py --config configs/cnn_lite.yaml --folds 3 --seeds 42   # trial
"""

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from python.experiment import Tee, load_config, make_results_dir  # noqa: E402
from python.models.cnn1d import ECGCNN, ModelConfig, n_params  # noqa: E402
from python.data.mitbih import REPO_ROOT  # noqa: E402
from python.models.cv_data import attach_extra_train, load_cv_data, with_extra_train  # noqa: E402
from python.models.metrics import CLASSES, format_report, mean_std, report  # noqa: E402
from python.models.training import (  # noqa: E402
    TrainConfig, class_weights, fit, fold_indices, predict_proba, rr_stats, set_seed,
)


def plot_history(history: list[dict], title: str, path: Path) -> None:
    ep = [h["epoch"] for h in history]
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.plot(ep, [h["train_loss"] for h in history], label="train (weighted CE)")
    ax.plot(ep, [h["val_loss"] for h in history], label="inner-val (weighted CE)")
    best = int(np.argmin([h["val_loss"] for h in history]))
    ax.axvline(ep[best], color="gray", ls="--", lw=0.8, label=f"best epoch {ep[best]}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--folds", default=None, help="comma list, default: config")
    ap.add_argument("--seeds", default=None, help="comma list, default: config")
    ap.add_argument("--tag", default="", help="run label appended to the results dir, e.g. A0")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--resume", action="store_true",
                    help="skip (seed, fold) runs whose final + snapshot checkpoints exist (after an interruption)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if cfg.get("dataset") != "DS1":
        raise SystemExit("CV runs on DS1 only")
    folds_run = [int(f) for f in args.folds.split(",")] if args.folds else cfg["folds"]
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else cfg["seeds"]
    mcfg, tcfg = ModelConfig.from_dict(cfg["model"]), TrainConfig.from_dict(cfg["train"])

    cv = load_cv_data(cfg["beats_config"])
    # re-normalisation inside augmentation must use the pipeline's beat_norm (single source: beats config)
    tcfg = dataclasses.replace(tcfg, norm_eps=cv.norm_eps, norm_clip=cv.norm_clip)
    d = cv.data
    if not tcfg.augment:
        d.pop("X_wide", None)      # only needed for R-jitter augmentation; saves memory (~4 bytes x 370 per beat)
    if d["rr"].shape[1] != mcfg.n_rr:
        raise SystemExit(f"model.n_rr = {mcfg.n_rr} but the beat cache has {d['rr'].shape[1]} RR features")
    # optional training-only extra beats (A5: SVDB) appended AFTER the MIT-BIH beats; they only ever join
    # the inner-train indices (fold splits, inner-val and OOF use the MIT-BIH part [:n_main] only)
    d, n_main, extra_idx, extra_name = attach_extra_train(cfg, d)
    y_main, rec_main, rpos_main = d["y"][:n_main], d["rec_id"][:n_main], d["r_pos"][:n_main]
    partial = sorted(folds_run) != [0, 1, 2, 3, 4] or seeds != cfg["seeds"]
    suffix = f"_{cfg['name']}" + (f"_trial-f{'-'.join(map(str, folds_run))}-s{'-'.join(map(str, seeds))}"
                                  if partial else "") + (f"_{args.tag}" if args.tag else "")
    out = make_results_dir(cfg | {"_cache": cv.cache.name} | ({"_extra_cache": extra_name} if extra_name else {}),
                           suffix=suffix)
    (out / "ckpt").mkdir(exist_ok=True)
    log = Tee(out / "log.txt")
    log(f"config {args.config} -> {out}")
    log(f"torch {torch.__version__}, CUDA {torch.version.cuda}, device {args.device}"
        + (f" ({torch.cuda.get_device_name(0)})" if args.device.startswith("cuda") else ""))
    log(f"cache {cv.cache} : {n_main} beats " +
        ", ".join(f"{c} {int(np.sum(y_main == i))}" for i, c in enumerate(CLASSES)))
    if extra_name:
        ye = d["y"][n_main:]
        log(f"extra training-only beats ({cfg['extra_train']}) from {extra_name}: {len(ye)} beats " +
            ", ".join(f"{c} {int(np.sum(ye == i))}" for i, c in enumerate(CLASSES)) + " -> inner-train only")
    log(f"model params: {n_params(ECGCNN(mcfg))}")

    seed_reports, fold_reports = [], {}
    snap = tcfg.snapshot_epochs
    if snap and (tcfg.early_stopping or max(snap) > tcfg.max_epochs):
        raise SystemExit("snapshot_epochs require early_stopping: false and max(snapshot) <= max_epochs")
    snap_reports = {e: [] for e in snap}
    for seed in seeds:
        oof = np.full((n_main, len(CLASSES)), np.nan, dtype=np.float32)
        oof_snap = {e: np.full_like(oof, np.nan) for e in snap}
        fold_of = np.full(n_main, -1, dtype=np.int8)
        for k in folds_run:
            idx = fold_indices(rec_main, rpos_main, cv.rec_len, cv.folds, k, cv.delay, cv.half, tcfg, cv.fs)
            idx = with_extra_train(idx, extra_idx, n_main)
            ytr = d["y"][idx["train"]]
            w = class_weights(ytr, tcfg.class_weight)
            mu, sd = rr_stats(d["rr"][idx["train"]])
            log(f"\n[seed {seed} fold {k}] outer {len(idx['outer'])}  inner-train {len(idx['train'])}  "
                f"inner-val {len(idx['inner_val'])}  buffer {len(idx['buffer'])}")
            log("  inner-train per class " + ", ".join(f"{c} {int(np.sum(ytr == i))}" for i, c in enumerate(CLASSES))
                + " | inner-val " + ", ".join(f"{c} {int(np.sum(d['y'][idx['inner_val']] == i))}"
                                              for i, c in enumerate(CLASSES)))
            log(f"  class weights {np.round(w, 3).tolist()}  rr mu {np.round(mu, 3).tolist()} "
                f"sigma {np.round(sd, 3).tolist()}")
            y_outer = d["y"][idx["outer"]]
            ck_final = out / "ckpt" / f"s{seed}_f{k}.pt"
            ck_snaps = [out / "ckpt" / f"s{seed}_f{k}_e{e}.pt" for e in snap]
            if args.resume and ck_final.exists() and all(p.exists() for p in ck_snaps):
                # RESUME after an interruption: this (seed, fold) run had finished; its OOF predictions are
                # recomputed from the saved weights (training is deterministic: equal to a re-train)
                def _load(p):
                    c = torch.load(p, weights_only=False)
                    m_ = ECGCNN(mcfg)
                    m_.load_state_dict(c["state_dict"])
                    return m_, c
                m_final, c_final = _load(ck_final)
                hist = c_final["history"]
                for e, p in zip(snap, ck_snaps):
                    oof_snap[e][idx["outer"]] = predict_proba(_load(p)[0], d, idx["outer"], args.device)
                probs = predict_proba(m_final, d, idx["outer"], args.device)
                oof[idx["outer"]] = probs
                fold_of[idx["outer"]] = k
                rep = report(y_outer, probs.argmax(1))
                fold_reports[f"s{seed}_f{k}"] = rep | {
                    "stop_epoch": len(hist), "best_epoch": int(np.argmin([h["val_loss"] for h in hist])) + 1,
                    "train_time_s": float("nan"), "mean_epoch_s": float(np.mean([h["time_s"] for h in hist])),
                    "resumed": True}
                log(f"  RESUMED from {ck_final.name} (+{len(ck_snaps)} snapshot checkpoints): no re-training")
                continue
            set_seed(seed)
            model = ECGCNN(mcfg)
            model.set_rr_stats(mu, sd)

            def diag(m, epoch, outer=idx["outer"], y_outer=y_outer):
                # DIAGNOSTIC ONLY: outer-fold metrics per epoch for the log/plots. Never used by
                # early stopping, the LR scheduler or checkpoint selection (those use inner-val loss).
                # With snapshot_epochs (fixed-epoch ablation) the outer predictions at those epochs
                # are stored: they ARE the OOF predictions of "train exactly e epochs".
                p = predict_proba(m, d, outer, args.device)
                if epoch in snap:
                    oof_snap[epoch][outer] = p
                    # keep the weights of every grid epoch (no re-training needed later)
                    torch.save({"state_dict": {kk: v.detach().cpu().clone() for kk, v in m.state_dict().items()},
                                "model": mcfg.to_dict(), "train": cfg["train"], "epoch": epoch,
                                "class_weights": w, "rr_mean": mu, "rr_std": sd, "seed": seed, "fold": k},
                               out / "ckpt" / f"s{seed}_f{k}_e{epoch}.pt")
                r = report(y_outer, p.argmax(1))
                rec = {c: r["per_class"][c]["recall"] for c in CLASSES}
                return {"outer_macro_f1": r["macro_f1"], "outer_recall": rec,
                        "msg": f"outer F1 {r['macro_f1']:.3f} R " + "/".join(f"{rec[c]:.2f}" for c in CLASSES)}

            t0 = time.time()
            hist = fit(model, d, idx["train"], idx["inner_val"], tcfg, w, seed, args.device,
                       log=lambda s: log(s), diag=diag)
            dt = time.time() - t0
            probs = predict_proba(model, d, idx["outer"], args.device)
            oof[idx["outer"]] = probs
            fold_of[idx["outer"]] = k
            rep = report(y_outer, probs.argmax(1))
            best_epoch = int(np.argmin([h["val_loss"] for h in hist])) + 1
            fold_reports[f"s{seed}_f{k}"] = rep | {"stop_epoch": len(hist), "best_epoch": best_epoch,
                                                   "train_time_s": dt,
                                                   "mean_epoch_s": float(np.mean([h["time_s"] for h in hist]))}
            log(f"  done: stopped at epoch {len(hist)} (best epoch {best_epoch}), {dt:.1f} s "
                f"({np.mean([h['time_s'] for h in hist]):.2f} s/epoch, train + inner-val only, diag excluded); "
                f"best inner-val loss {min(h['val_loss'] for h in hist):.4f}")
            log("  outer fold (OOF) " + format_report(rep).replace("\n", "\n  "))
            torch.save({"state_dict": model.state_dict(), "model": mcfg.to_dict(), "train": cfg["train"],
                        "class_weights": w, "rr_mean": mu, "rr_std": sd, "seed": seed, "fold": k,
                        "history": hist}, out / "ckpt" / f"s{seed}_f{k}.pt")
            (out / f"history_s{seed}_f{k}.json").write_text(json.dumps(hist, indent=1), encoding="utf-8")
            plot_history(hist, f"{cfg['name']} seed {seed} fold {k}", out / f"curve_s{seed}_f{k}.png")

        done = fold_of >= 0
        np.savez_compressed(out / f"oof_s{seed}.npz", probs=oof, fold=fold_of, y=y_main,
                            rec_id=rec_main, r_pos=rpos_main)
        if np.all(done):
            rep = report(y_main, oof.argmax(1))
            seed_reports.append(rep)
            log(f"\n=== seed {seed}: pooled OOF over 5 folds ===\n" + format_report(rep))
            for e in snap:
                np.savez_compressed(out / f"oof_s{seed}_e{e}.npz", probs=oof_snap[e], fold=fold_of, y=y_main,
                                    rec_id=rec_main, r_pos=rpos_main)
                rep_e = report(y_main, oof_snap[e].argmax(1))
                snap_reports[e].append(rep_e)
                log(f"--- seed {seed}, fixed {e} epochs: pooled OOF ---\n" + format_report(rep_e))

    summary = {"fold_reports": fold_reports, "seed_reports": seed_reports}
    if snap and all(len(v) == len(seeds) for v in snap_reports.values()):
        summary["snapshots"] = {str(e): {"seed_reports": v, "mean_std": mean_std(v)} for e, v in snap_reports.items()}
        log("\n=== fixed-epoch snapshots: pooled OOF macro-F1 (mean ± std over seeds) ===")
        for e, v in snap_reports.items():
            ms = mean_std(v)
            log(f"  {e:>3} epochs: macro-F1 {ms['macro_f1']['mean']:.4f} ± {ms['macro_f1']['std']:.4f}  "
                f"per seed {[round(r['macro_f1'], 4) for r in v]}  N->F {ms['n_pred_N_as_F']['mean']:.0f}")
    if seed_reports:
        summary["mean_std"] = mean_std(seed_reports)
        stops = [v["stop_epoch"] for v in fold_reports.values()]
        summary["stop_epoch"] = {"mean": float(np.mean(stops)), "std": float(np.std(stops, ddof=1)),
                                 "min": int(min(stops)), "max": int(max(stops))}
        log(f"stop epoch: mean {np.mean(stops):.1f} ± {np.std(stops, ddof=1):.1f} (min {min(stops)}, max {max(stops)})")
        ms = summary["mean_std"]
        log(f"\n=== {cfg['name']}: OOF macro-F1 {ms['macro_f1']['mean']:.4f} ± {ms['macro_f1']['std']:.4f} "
            f"(n seeds = {ms['macro_f1']['n']}) ===")
        for c in CLASSES:
            log(f"  {c}: " + "  ".join(f"{m} {ms[c][m]['mean']:.4f}±{ms[c][m]['std']:.4f}"
                                       for m in ("precision", "recall", "f1")))
        log(f"  macro-F1 N/S/V {ms['macro_f1_nsv']['mean']:.4f}±{ms['macro_f1_nsv']['std']:.4f}; " +
            "; ".join(f"{g} Se {ms['ec57'][g]['se']['mean']:.4f}±{ms['ec57'][g]['se']['std']:.4f} "
                      f"+P {ms['ec57'][g]['ppv']['mean']:.4f}±{ms['ec57'][g]['ppv']['std']:.4f}"
                      for g in ("SVEB", "VEB")))
    (out / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(f"\nsaved -> {out}")
    log.close()


if __name__ == "__main__":
    main()
