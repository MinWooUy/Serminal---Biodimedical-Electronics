"""Quantise the final Lite model (int8) and run an EXPORT SANITY CHECK of the fully integer pipeline.

Calibration: max|x| on 4000 beats of the final model's inner-train (DS1 + SVDB n25), seed 42.
Sanity check (NOT a reported metric; these beats drove the LR scheduler): on the final inner-val
(last 20% of every DS1 record) compare the float pipeline + float model with the fully integer pipeline
(integer filter / refinement / z-score / RR + int8 CNN). Writes lite_int8.npz next to model.pt.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from python.data.mitbih import REPO_ROOT  # noqa: E402
from python.experiment import Tee, load_config  # noqa: E402
from python.models.cnn1d import ECGCNN, ModelConfig  # noqa: E402
from python.models.cv_data import attach_extra_train, load_cv_data, with_extra_train  # noqa: E402
from python.models.metrics import format_report, report  # noqa: E402
from python.models.training import TrainConfig, fold_indices  # noqa: E402
from python.quant.int8 import float_forward, fold_model, int_forward_q, quantize_model, save_quant_model  # noqa: E402
from scripts.pipeline_int_eval import integer_beats, quantise_rr_parts  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", required=True, help="results dir of the final Lite (model.pt, config.yaml)")
    ap.add_argument("--n-calib", type=int, default=4000)
    args = ap.parse_args()
    out = Path(args.final)
    cfg = load_config(out / "config.yaml")
    ck = torch.load(out / "model.pt", weights_only=False)
    m = ECGCNN(ModelConfig.from_dict(ck["model"]))
    m.load_state_dict(ck["state_dict"])
    bcfg = load_config(REPO_ROOT / cfg["beats_config"])
    det_cfg = load_config(REPO_ROOT / bcfg["detector_config"])
    cv = load_cv_data(cfg["beats_config"])
    log = Tee(out / "quantize.log")
    ib = integer_beats(cv, det_cfg, bcfg, lambda s: None)          # integer pre-processing of all DS1 train/val beats
    d, n_main, extra_idx, extra_name = attach_extra_train(cfg, cv.data)
    tcfg = TrainConfig.from_dict(cfg["train"])
    idx = fold_indices(d["rec_id"][:n_main], d["r_pos"][:n_main], cv.rec_len, [[]], 0, cv.delay, cv.half, tcfg, cv.fs)
    idx = with_extra_train(idx, extra_idx, n_main)
    if {k: len(v) for k, v in idx.items()} != ck["inner_split"]:
        raise SystemExit("inner split differs from the one used in final training")
    rng = np.random.default_rng(int(ck["seed"]))
    cal = np.sort(rng.choice(idx["train"], size=min(args.n_calib, len(idx["train"])), replace=False))
    fm = fold_model(m)
    qm = quantize_model(fm, d["X"][cal], d["rr"][cal], "max")
    save_quant_model(qm, out / "lite_int8.npz")
    log(f"quantised {out.name}: calibration max|x| on {len(cal)} inner-train beats (incl. {extra_name}); "
        f"conv (a_out, k min..max) {[(c['a_out'], int(c['k'].min()), int(c['k'].max())) for c in qm.convs]}, "
        f"rr_exp {qm.rr_exp.tolist()}, fc1 a_out {qm.fc1['a_out']}, fc2 k {qm.fc2['k']}")
    iv = idx["inner_val"]
    y = d["y"][iv]
    p_float = float_forward(fm, d["X"][iv], d["rr"][iv]).argmax(1)
    p_int = int_forward_q(qm, ib["z_q"][iv], quantise_rr_parts(ib["rr_parts"][iv], qm.rr_exp)).argmax(1)
    rf, ri = report(y, p_float), report(y, p_int)
    log(f"\nEXPORT SANITY CHECK on the final inner-val ({len(iv)} DS1 beats; NOT a reported metric):")
    log(f"float pipeline + float model:\n{format_report(rf)}")
    log(f"fully integer pipeline (int filter / z / RR + int8 CNN):\n{format_report(ri)}")
    log(f"argmax agreement {np.mean(p_float == p_int):.4f}; macro-F1 float {rf['macro_f1']:.4f} vs integer {ri['macro_f1']:.4f}")
    (out / "quantize.json").write_text(json.dumps({"float": rf, "integer": ri, "agreement": float(np.mean(p_float == p_int)),
                                                   "n_calib": int(len(cal))}, indent=2), encoding="utf-8")
    log(f"saved -> {out / 'lite_int8.npz'}")
    log.close()


if __name__ == "__main__":
    main()
