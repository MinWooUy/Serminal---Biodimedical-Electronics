"""Beat-by-beat QRS detector evaluation (EC57-style).

- Reference = all beat annotations (Q included); annotations and detections
  inside ventricular flutter segments are excluded before matching.
- One-to-one matching: all (ref, det) pairs with |delta| <= tol are sorted by
  (|delta|, ref index, det index) and assigned greedily; each ref and each det
  is used at most once. Tolerance 150 ms means |delta| <= 150 ms (54 samples
  at 360 Hz); 75 ms (27 samples) is reported as a stricter reference.
"""

from dataclasses import dataclass

import numpy as np


def tol_samples(tol_ms: float, fs: int) -> int:
    return int(round(tol_ms * fs / 1000.0))


def outside_vf(pos: np.ndarray, vf_segments: np.ndarray) -> np.ndarray:
    """Boolean mask: True where pos is not inside any [start, end] VF segment."""
    pos = np.asarray(pos, dtype=np.int64)
    keep = np.ones(len(pos), dtype=bool)
    for start, end in np.asarray(vf_segments).reshape(-1, 2):
        keep &= ~((pos >= start) & (pos <= end))
    return keep


@dataclass
class MatchResult:
    pairs: np.ndarray        # (n_tp, 2) indices into (ref, det)
    fn: np.ndarray           # unmatched ref indices
    fp: np.ndarray           # unmatched det indices

    @property
    def tp(self) -> int:
        return len(self.pairs)


def match_beats(ref: np.ndarray, det: np.ndarray, tol: int) -> MatchResult:
    ref = np.asarray(ref, dtype=np.int64)
    det = np.asarray(det, dtype=np.int64)
    order = np.argsort(det, kind="stable")
    det_sorted = det[order]
    cand = []
    for i, r in enumerate(ref):
        lo = np.searchsorted(det_sorted, r - tol, side="left")
        hi = np.searchsorted(det_sorted, r + tol, side="right")
        for j in order[lo:hi]:
            cand.append((abs(int(det[j]) - int(r)), i, int(j)))
    cand.sort()
    used_r = np.zeros(len(ref), dtype=bool)
    used_d = np.zeros(len(det), dtype=bool)
    pairs = []
    for _, i, j in cand:
        if not used_r[i] and not used_d[j]:
            used_r[i] = used_d[j] = True
            pairs.append((i, j))
    pairs = np.array(sorted(pairs), dtype=np.int64).reshape(-1, 2)
    return MatchResult(pairs, np.flatnonzero(~used_r), np.flatnonzero(~used_d))


def pair_errors_ms(ref_pos: np.ndarray, det_pos: np.ndarray, pairs: np.ndarray,
                   fs: int) -> np.ndarray:
    """Signed error det - ref (ms) for each matched pair."""
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    return (np.asarray(det_pos)[pairs[:, 1]] - np.asarray(ref_pos)[pairs[:, 0]]) * 1000.0 / fs


def summarize_errors(err_ms: np.ndarray) -> dict[str, float]:
    if len(err_ms) == 0:
        return {"mean_ms": float("nan"), "std_ms": float("nan"), "p95_abs_ms": float("nan"), "n": 0}
    return {"mean_ms": float(err_ms.mean()), "std_ms": float(err_ms.std()),
            "p95_abs_ms": float(np.percentile(np.abs(err_ms), 95)), "n": int(len(err_ms))}


def position_errors(ref_pos: np.ndarray, det_pos: np.ndarray, pairs: np.ndarray,
                    fs: int) -> dict[str, float]:
    """Signed error det - ref over matched pairs, in ms: mean, std, p95 of |error|, n.

    Single implementation for every position-error figure (raw PT, refined, re-aligned).
    """
    return summarize_errors(pair_errors_ms(ref_pos, det_pos, pairs, fs))


def se_ppv(tp: int, fn: int, fp: int) -> tuple[float, float]:
    se = tp / (tp + fn) if tp + fn else float("nan")
    ppv = tp / (tp + fp) if tp + fp else float("nan")
    return se, ppv


def evaluate_record(ann: np.ndarray, det_refined: np.ndarray, det_coarse: np.ndarray,
                    ann_realigned: np.ndarray, vf_segments: np.ndarray, fs: int,
                    tols_ms: tuple[float, ...] = (150.0, 75.0)) -> dict:
    """Se/+P use refined detections vs original annotations.

    Position errors: coarse and refined detections vs original annotations, and
    refined detections vs re-aligned annotations (pairs from the 150 ms match).
    """
    ka = outside_vf(ann, vf_segments)
    kd = outside_vf(det_refined, vf_segments)
    ann, ann_realigned = np.asarray(ann)[ka], np.asarray(ann_realigned)[ka]
    det_refined, det_coarse = np.asarray(det_refined)[kd], np.asarray(det_coarse)[kd]

    out = {"n_ref": int(len(ann)), "n_det": int(len(det_refined)),
           # kept (non-VF) arrays, for pooling and per-beat analysis; not JSON
           "arrays": {"ann_idx": np.flatnonzero(ka), "det_idx": np.flatnonzero(kd),
                      "ann": ann, "ann_realigned": ann_realigned,
                      "det_refined": det_refined, "det_coarse": det_coarse}}
    for tol_ms in tols_ms:
        m = match_beats(ann, det_refined, tol_samples(tol_ms, fs))
        se, ppv = se_ppv(m.tp, len(m.fn), len(m.fp))
        key = f"{int(tol_ms)}ms"
        out[key] = {"tp": m.tp, "fn": int(len(m.fn)), "fp": int(len(m.fp)), "se": se, "ppv": ppv}
        if tol_ms == tols_ms[0]:
            out["match"] = m
            out["err_coarse_vs_ann"] = position_errors(ann, det_coarse, m.pairs, fs)
            out["err_refined_vs_ann"] = position_errors(ann, det_refined, m.pairs, fs)
            out["err_refined_vs_realigned"] = position_errors(ann_realigned, det_refined, m.pairs, fs)
    return out
