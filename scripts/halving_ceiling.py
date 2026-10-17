"""Ceiling for a 'replace beats accepted only thanks to threshold halving' rule (DS1, counting only).

A detection is halving-dependent if, when accepted, the irregular flag was set
and it would NOT have passed the un-halved thresholds (THR1 for regular
decisions, THR1 >> 1 for search-back). Only those FPs can be touched by such a
rule. Prints the +P ceiling if all of them were removed (and, generously, if the
FNs < 200 ms after them became TPs).
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from python.data.mitbih import load_record  # noqa: E402
from python.data.splits import get_ds1_records  # noqa: E402
from python.dsp import pan_tompkins as pt  # noqa: E402
from python.dsp.evaluate_detector import evaluate_record  # noqa: E402
from python.dsp.realign import half_window_samples, realign  # noqa: E402
from python.experiment import load_config  # noqa: E402


class TaggedEngine(pt.DecisionEngine):
    """DecisionEngine that records, per accepted beat, whether halving was needed."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.tags: list[bool] = []
        self._snap = (0, 0)

    def _full(self):
        return (self.npki + ((self.spki - self.npki) >> 2), self.npkf + ((self.spkf - self.npkf) >> 2))

    def _classify(self, pk, out):
        self._snap = self._full()
        super()._classify(pk, out)

    def _searchback(self, out):
        i, f = self._full()
        self._snap = (i >> 1, f >> 1)
        super()._searchback(out)

    def _accept(self, pk, searchback, out):
        si, sf = self._snap
        self.tags.append(bool(self.irregular and not (pk.mwi > si and pk.bp > sf)))
        super()._accept(pk, searchback, out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/detector_pt.yaml")
    args = ap.parse_args()
    cfg, sos = pt.detector_from_config(load_config(args.config))
    if not cfg.irregular_halving:
        raise SystemExit("ceiling is only meaningful with irregular_halving = true")
    half = half_window_samples(cfg.fs)
    tot = np.zeros(3, dtype=np.int64)
    removable, recoverable, per_rec = 0, 0, {}
    for rid in get_ds1_records():
        rec = load_record(rid)
        res = pt.detect_offline(rec.adc, rec.adc_zero, rec.adc_gain, cfg, sos)
        st = pt.frontend_offline(rec.adc, rec.adc_zero, cfg)
        eng = TaggedEngine(cfg)
        for h, d, m in zip(st["hp"].tolist(), st["d"].tolist(), st["mwi"].tolist()):
            eng.process(h, d, m)
        eng.flush()
        assert len(eng.tags) == len(res.detections)          # same deterministic run
        tags = np.array(eng.tags)
        ann_re, _ = realign(rec.r_samples, res.x_filt, res.delay, half, cfg.realign_mode)
        ev = evaluate_record(rec.r_samples, res.r_refined, res.r_coarse, ann_re, rec.vf_segments, cfg.fs)
        e, arr, m = ev["150ms"], ev["arrays"], ev["match"]
        tot += [e["tp"], e["fp"], e["fn"]]
        fp_tag = tags[arr["det_idx"][m.fp]]
        fp_pos = arr["det_refined"][m.fp][fp_tag]
        n_rec = int(sum(np.any((f - fp_pos > 0) & (f - fp_pos < cfg.refractory)) for f in arr["ann"][m.fn]))
        removable += int(fp_tag.sum())
        recoverable += n_rec
        per_rec[rid] = (e["fp"], int(fp_tag.sum()), n_rec)
    tp, fp, fn = (int(v) for v in tot)
    print(f"DS1 total: TP {tp}  FP {fp}  FN {fn}  Se {100 * tp / (tp + fn):.3f}%  +P {100 * tp / (tp + fp):.3f}%")
    print(f"108: FP {per_rec[108][0]}, halving-dependent {per_rec[108][1]}, FN recoverable {per_rec[108][2]}")
    print("halving-dependent FP per record:", {k: v[1] for k, v in per_rec.items() if v[1]})
    print(f"max removable FP on DS1: {removable}")
    print(f"ceiling +P (remove all):            {100 * tp / (tp + fp - removable):.3f}%")
    print(f"ceiling +P (+{recoverable} FN -> TP, generous): "
          f"{100 * (tp + recoverable) / (tp + recoverable + fp - removable):.3f}%")
    need = fp - int(tp * 0.005 / 0.995)
    print(f"FP to remove for +P >= 99.5%: >= {need}")


if __name__ == "__main__":
    main()
