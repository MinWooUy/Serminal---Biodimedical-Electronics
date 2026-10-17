"""Final model: train on ALL DS1 records with the chosen base configuration (no outer fold, no OOF).

Same trajectory as the CV runs: every DS1 record is split in time (first 80% inner-train, last 20%
inner-val, 1 s buffer); ReduceLROnPlateau on the weighted inner-val loss; no early stopping; the LAST
weights after the chosen number of epochs are kept. Seed 42.

  python scripts/train_final.py --config configs/cnn_full.yaml --epochs 8                      # Full: sqrt_inverse CE @8
  python scripts/train_final.py --config configs/cnn_lite.yaml --epochs 12 --loss focal        # Lite: focal(2)+sqrt_inv @12

Writes results/<date>_<hash>_final_<name>/: config.yaml, model.pt (state_dict + RR mu/sigma +
class weights + configs + cache name), history.json, curve.png, log.txt. DS2 is never touched.
"""

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from python.data.constants import DS1  # noqa: E402
from python.experiment import Tee, load_config, make_results_dir  # noqa: E402
from python.models.cnn1d import ECGCNN, ModelConfig, n_params  # noqa: E402
from python.models.cv_data import attach_extra_train, load_cv_data, with_extra_train  # noqa: E402
from python.models.metrics import CLASSES  # noqa: E402
from python.models.training import (  # noqa: E402
    TrainConfig, class_weights, fit, fold_indices, rr_stats, set_seed,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="base CNN config (model + train)")
    ap.add_argument("--epochs", type=int, required=True, help="chosen number of epochs")
    ap.add_argument("--loss", default=None, help="override train.loss (e.g. focal for Lite)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if cfg.get("dataset") != "DS1":
        raise SystemExit("final training uses DS1 only")
    cfg["train"] = {**cfg["train"], "max_epochs": args.epochs, "early_stopping": False}
    cfg["train"].pop("snapshot_epochs", None)
    if args.loss:
        cfg["train"]["loss"] = args.loss
    cfg["seeds"], cfg["folds"] = [args.seed], []
    mcfg, tcfg = ModelConfig.from_dict(cfg["model"]), TrainConfig.from_dict(cfg["train"])

    cv = load_cv_data(cfg["beats_config"])
    tcfg = dataclasses.replace(tcfg, norm_eps=cv.norm_eps, norm_clip=cv.norm_clip)
    d = cv.data
    if d["rr"].shape[1] != mcfg.n_rr:
        raise SystemExit("model.n_rr does not match the beat cache")
    if not tcfg.augment:
        d.pop("X_wide", None)
    # training-only extra beats (Lite: SVDB n25, same sample seed as in CV) -> inner-train only
    d, n_main, extra_idx, extra_name = attach_extra_train(cfg, d)
    out = make_results_dir(cfg | {"_cache": cv.cache.name} | ({"_extra_cache": extra_name} if extra_name else {}),
                           suffix=f"_final_{cfg['name']}")
    log = Tee(out / "log.txt")
    log(f"final training on all DS1 ({len(DS1)} records), seed {args.seed}, {args.epochs} epochs, "
        f"loss {tcfg.loss}, class weights {tcfg.class_weight} -> {out}")
    # one 'fold' with an empty outer set: every DS1 record is time-split into inner-train / inner-val
    idx = fold_indices(d["rec_id"][:n_main], d["r_pos"][:n_main], cv.rec_len, [[]], 0, cv.delay, cv.half, tcfg, cv.fs)
    assert len(idx["outer"]) == 0
    idx = with_extra_train(idx, extra_idx, n_main)
    if extra_name:
        log(f"extra training-only beats {cfg['extra_train']} from {extra_name}: {len(extra_idx)} -> inner-train only")
    ytr = d["y"][idx["train"]]
    w = class_weights(ytr, tcfg.class_weight)
    mu, sd = rr_stats(d["rr"][idx["train"]])
    log(f"inner-train {len(idx['train'])} beats (" + ", ".join(f"{c} {int(np.sum(ytr == i))}"
                                                           for i, c in enumerate(CLASSES))
        + f"), inner-val {len(idx['inner_val'])}, buffer {len(idx['buffer'])}")
    log(f"class weights {np.round(w, 4).tolist()}; RR mu {np.round(mu, 4).tolist()} sigma {np.round(sd, 4).tolist()}")
    set_seed(args.seed)
    model = ECGCNN(mcfg)
    model.set_rr_stats(mu, sd)
    log(f"params {n_params(model)}")
    hist = fit(model, d, idx["train"], idx["inner_val"], tcfg, w, args.seed, args.device, log=lambda s: log(s))
    torch.save({"state_dict": model.state_dict(), "model": mcfg.to_dict(), "train": cfg["train"],
                "class_weights": w, "rr_mean": mu, "rr_std": sd, "seed": args.seed, "epochs": args.epochs,
                "cache": cv.cache.name, "beats_config": cfg["beats_config"],
                "extra_train": cfg.get("extra_train"), "extra_cache": extra_name, "n_main": n_main,
                "inner_split": {k: len(v) for k, v in idx.items()}, "history": hist}, out / "model.pt")
    (out / "history.json").write_text(json.dumps(hist, indent=1), encoding="utf-8")
    from scripts.train_cv import plot_history
    plot_history(hist, f"final {cfg['name']} seed {args.seed}", out / "curve.png")
    log(f"saved -> {out / 'model.pt'} (no held-out data: this run has no OOF metrics)")
    log.close()


if __name__ == "__main__":
    main()
