"""Beat dataset: 360-sample windows around R + 4 RR features, cached per record.

R positions come from `rr_source`:
  detector   - refined R from the integer Pan-Tompkins (default; same as deployment)
  annotation - annotation positions re-aligned with the same rule (realign_mode)

Pipeline per record (identical for both sources and for streaming):
  1. RR features on the FULL R sequence (Q beats, FPs and VF detections included,
     because a deployed detector sees them too):
       rr_prev = R[i] - R[i-1],  rr_next = R[i+1] - R[i],
       (each RR interval clipped to rr_clip_s = [0.2, 3.0] s first)
       rr_avg  = mean of the past RRs available up to and including rr_prev
                 (at most rr_avg_n = 8, at least 1; never a future RR),
       ratio   = rr_prev / rr_avg.            (all in seconds)
  2. Drop the first beat (no rr_prev) and the last beat (no rr_next).
  3. Drop beats whose window is incomplete: the 0.5-40 Hz filtered signal lags by
     `delay` samples, so the window is x_f[c-180 : c+180] with c = R + delay
     (R at index 180 of the 360-sample window).
  4. Per-beat z-score (beat_norm): (w - mean) / (std + eps), clipped to [-clip, clip].
  5. Labels: detector beats are matched one-to-one with annotations within
     +/-150 ms (same matcher as the detector evaluation); a matched beat takes
     the AAMI class of its annotation. Unmatched detections (FP) and detections in
     VF segments are kept in the cache with kind != 0 for end-to-end evaluation
     but never enter train/val. Q beats (y = -1) are dropped only at load time,
     i.e. after the RR features were computed with them.

Deployment latency per beat = rr_next (wait for the next R) + detector latency;
the window's 180 samples after R are always covered by that wait.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from python.data.constants import DS1, Q_LABEL
from python.data.mitbih import REPO_ROOT, Record
from python.dsp.evaluate_detector import match_beats, outside_vf, tol_samples
from python.dsp.filters import apply_offline, group_delay_compensation
from python.dsp.pan_tompkins import PTConfig, detect_offline
from python.dsp.realign import half_window_samples, realign

BEATS_VERSION = "beats-v2"   # v2: + X_wide (370-sample un-normalised windows for R-jitter augmentation)
WIDE_PAD = 5                  # max R jitter (samples)
CACHE_ROOT = REPO_ROOT / "data" / "cache"
RR_SOURCES = ("detector", "annotation")

KIND_LABELED = 0     # matched to an annotation (may still be Q)
KIND_FP = 1          # detection without annotation (outside VF)
KIND_VF = 2          # detection inside a VF segment
Y_UNLABELED = -2

RR_FEATURES = ("rr_prev", "rr_next", "rr_avg8", "rr_ratio")
RR_FEATURES_EXT = RR_FEATURES + ("rr_prev_over_next", "rr_prev_over_avg32")   # A4: rr_avg_long_n > 0


@dataclass(frozen=True)
class BeatConfig:
    rr_source: str = "detector"
    half_window: int = 180
    match_tol_ms: float = 150.0
    rr_avg_n: int = 8
    norm_type: str = "zscore"
    norm_eps: float = 1e-6
    norm_clip: float = 5.0
    rr_clip_s: tuple[float, float] = (0.2, 3.0)
    rr_avg_long_n: int = 0          # > 0: add rr_prev/rr_next and rr_prev/rr_avg_long (6 features)

    @classmethod
    def from_dict(cls, d: dict) -> "BeatConfig":
        norm = d.get("beat_norm", {})
        return cls(rr_source=d["rr_source"], half_window=d["half_window"],
                   match_tol_ms=d["match_tol_ms"], rr_avg_n=d["rr_avg_n"],
                   norm_type=norm.get("type", "zscore"), norm_eps=float(norm.get("eps", 1e-6)),
                   norm_clip=float(norm.get("clip", 5.0)),
                   rr_clip_s=tuple(float(v) for v in d["rr_clip_s"]),
                   rr_avg_long_n=int(d.get("rr_avg_long_n", 0)))


# ---------------------------------------------------------------- RR features

def rr_clip_samples(rr_clip_s: tuple[float, float], fs: int) -> tuple[int, int]:
    """[0.2, 3.0] s -> [72, 1080] samples at 360 Hz (integer bounds, C-portable)."""
    lo, hi = int(round(rr_clip_s[0] * fs)), int(round(rr_clip_s[1] * fs))
    if not 0 < lo < hi:
        raise ValueError(f"bad RR clip {rr_clip_s}")
    return lo, hi


def rr_features(r_pos: np.ndarray, fs: int, avg_n: int = 8,
                rr_clip: tuple[int, int] | None = None, avg_long_n: int = 0) -> np.ndarray:
    """Offline RR features for beats 1..n-2 of r_pos -> array (n-2, 4) or (n-2, 6).

    4 features: rr_prev, rr_next, rr_avg (seconds), rr_prev / rr_avg.
    avg_long_n > 0 adds rr_prev / rr_next and rr_prev / rr_avg_long (mean of up to avg_long_n
    past RRs, incl. rr_prev, at least 1, never a future RR).

    Every RR interval is first clipped to rr_clip (samples), then rr_prev, rr_next,
    rr_avg and the ratio are computed from the clipped intervals. Row k corresponds
    to beat i = k + 1. Integer sums in samples, then one division, so RRStream
    (beat-by-beat) gives bit-identical values.
    """
    r = np.asarray(r_pos, dtype=np.int64)
    n_feat = 6 if avg_long_n > 0 else 4
    if len(r) < 3:
        return np.zeros((0, n_feat))
    rr = np.diff(r)                                   # rr[j] = r[j+1] - r[j]
    if rr_clip is not None:
        rr = np.clip(rr, rr_clip[0], rr_clip[1])
    out = np.empty((len(r) - 2, n_feat))
    for k, i in enumerate(range(1, len(r) - 1)):
        past = rr[max(0, i - avg_n):i]                # RRs ending at beats <= i
        rr_prev, rr_next = int(rr[i - 1]), int(rr[i])
        avg = int(past.sum()) / len(past)
        out[k, :4] = (rr_prev / fs, rr_next / fs, avg / fs, rr_prev / avg)
        if avg_long_n > 0:
            past_l = rr[max(0, i - avg_long_n):i]
            avg_l = int(past_l.sum()) / len(past_l)
            out[k, 4:] = (rr_prev / rr_next, rr_prev / avg_l)
    return out


class RRStream:
    """Beat-by-beat RR features: push(R) returns (index, features) of the previous beat
    once its next R is known (i.e. features of beat i are emitted when R[i+1] arrives).
    Each RR interval is clipped to rr_clip (samples) as soon as it is formed."""

    def __init__(self, fs: int, avg_n: int = 8, rr_clip: tuple[int, int] | None = None,
                 avg_long_n: int = 0):
        self.fs, self.avg_n, self.rr_clip, self.avg_long_n = fs, avg_n, rr_clip, avg_long_n
        self._keep = max(avg_n, avg_long_n)
        self._last_r: int | None = None
        self._prev_rr: int | None = None   # clipped RR ending at the current candidate beat
        self._past: list[int] = []         # last max(avg_n, avg_long_n) clipped RRs ending at the candidate beat
        self._n = 0

    def push(self, r: int):
        r = int(r)
        out = None
        if self._last_r is not None:
            rr = r - self._last_r
            if self.rr_clip is not None:
                rr = min(max(rr, self.rr_clip[0]), self.rr_clip[1])
            if self._prev_rr is not None:          # features of the previous beat are complete
                p = self._past[-self.avg_n:]
                avg = sum(p) / len(p)
                feats = (self._prev_rr / self.fs, rr / self.fs, avg / self.fs, self._prev_rr / avg)
                if self.avg_long_n > 0:
                    pl = self._past[-self.avg_long_n:]
                    feats += (self._prev_rr / rr, self._prev_rr / (sum(pl) / len(pl)))
                out = (self._n - 1, feats)
            self._prev_rr = rr                     # becomes rr_prev of the newest beat
            self._past.append(rr)
            if len(self._past) > self._keep:
                self._past.pop(0)
        self._last_r = r
        self._n += 1
        return out


# ---------------------------------------------------------------- windows

def extract_windows(x_filt: np.ndarray, r_pos: np.ndarray, delay: int, half: int
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Windows x_f[c-half : c+half], c = R + delay. Returns (windows (m, 2*half), kept mask)."""
    c = np.asarray(r_pos, dtype=np.int64) + delay
    keep = (c - half >= 0) & (c + half <= len(x_filt))
    idx = c[keep][:, None] + np.arange(-half, half)[None, :]
    return x_filt[idx], keep


def extract_wide(x_filt: np.ndarray, r_pos: np.ndarray, delay: int, half: int, pad: int) -> np.ndarray:
    """Un-normalised filtered windows x_f[c-half-pad : c+half+pad] (edge-padded at the record ends).
    TRAINING-ONLY input for R-jitter augmentation (crop of 2*half samples at offset pad+s, |s| <= pad);
    the centre crop (s = 0) is exactly the deployment window."""
    c0 = np.asarray(r_pos, dtype=np.int64) + delay
    if np.any(c0 - half < 0) or np.any(c0 + half > len(x_filt)):
        raise ValueError("extract_wide: beat without a complete centre window (should have been dropped)")
    xp = np.pad(x_filt, pad, mode="edge")
    c = c0 + pad
    idx = c[:, None] + np.arange(-half - pad, half + pad)[None, :]
    return xp[idx]


def normalize_beats(w: np.ndarray, eps: float, clip: float) -> np.ndarray:
    mu = w.mean(axis=1, keepdims=True)
    sd = w.std(axis=1, keepdims=True)
    return np.clip((w - mu) / (sd + eps), -clip, clip)


# ---------------------------------------------------------------- per record

def r_sequence(rec: Record, pt: PTConfig, sos: np.ndarray, source: str, match_tol_ms: float = 150.0):
    """Full R sequence for a record plus labels/kind, and the filtered signal.

    Returns (r_pos, y, kind, x_filt, delay, stats) where stats holds per-class FN
    counts (detector source) and the FP count.
    """
    if source not in RR_SOURCES:
        raise ValueError(f"unknown rr_source {source!r}")
    half_re = half_window_samples(pt.fs)
    if source == "detector":
        res = detect_offline(rec.adc, rec.adc_zero, rec.adc_gain, pt, sos)
        x_filt, delay = res.x_filt, res.delay
        r = res.r_refined
        ka = outside_vf(rec.r_samples, rec.vf_segments)
        kd = outside_vf(r, rec.vf_segments)
        ann_idx, det_idx = np.flatnonzero(ka), np.flatnonzero(kd)
        m = match_beats(rec.r_samples[ka], r[kd], tol_samples(match_tol_ms, pt.fs))
        y = np.full(len(r), Y_UNLABELED, dtype=np.int8)
        kind = np.full(len(r), KIND_VF, dtype=np.int8)
        kind[det_idx] = KIND_FP
        y[det_idx[m.pairs[:, 1]]] = rec.aami[ann_idx[m.pairs[:, 0]]]
        kind[det_idx[m.pairs[:, 1]]] = KIND_LABELED
        fn_cls = rec.aami[ann_idx[m.fn]]
        stats = {"fn": _class_counts(fn_cls), "fp": int(len(m.fp)),
                 "vf_det": int(np.sum(~kd)), "n_ann": _class_counts(rec.aami[ka])}
    else:
        x_filt = apply_offline(rec.signal, sos)
        delay = group_delay_compensation(sos, pt.fs)
        r, _ = realign(rec.r_samples, x_filt, delay, half_re, pt.realign_mode)
        y = rec.aami.astype(np.int8).copy()
        kind = np.full(len(r), KIND_LABELED, dtype=np.int8)
        stats = {"fn": _class_counts(np.array([], dtype=np.int8)), "fp": 0, "vf_det": 0,
                 "n_ann": _class_counts(rec.aami)}
    order = np.argsort(r, kind="stable")
    if np.any(np.diff(r[order]) <= 0):
        raise ValueError(f"record {rec.rec_id}: non-increasing R sequence")
    return r[order], y[order], kind[order], x_filt, delay, stats


def _class_counts(y: np.ndarray) -> dict[str, int]:
    names = {0: "N", 1: "S", 2: "V", 3: "F", Q_LABEL: "Q"}
    return {n: int(np.sum(y == k)) for k, n in names.items()}


def build_record(rec: Record, pt: PTConfig, sos: np.ndarray, bc: BeatConfig) -> tuple[dict, dict]:
    r, y, kind, x_filt, delay, stats = r_sequence(rec, pt, sos, bc.rr_source, bc.match_tol_ms)
    rr = rr_features(r, pt.fs, bc.rr_avg_n, rr_clip_samples(bc.rr_clip_s, pt.fs), bc.rr_avg_long_n)
    inner = slice(1, len(r) - 1)                       # step 2: drop first and last beat
    r_in, y_in, kind_in = r[inner], y[inner], kind[inner]
    win, keep = extract_windows(x_filt, r_in, delay, bc.half_window)
    if bc.norm_type != "zscore":
        raise ValueError(f"unknown beat_norm {bc.norm_type!r}")
    X = normalize_beats(win, bc.norm_eps, bc.norm_clip).astype(np.float32)

    lab = kind == KIND_LABELED
    first_last = np.zeros(len(r), dtype=bool)
    first_last[[0, -1]] = True
    edge = np.zeros(len(r), dtype=bool)
    edge[1:-1] = ~keep
    stats |= {
        "lost_first_last": _class_counts(y[lab & first_last]),
        "lost_edge": _class_counts(y[lab & edge]),
        "fp_kept": int(np.sum(kind_in[keep] == KIND_FP)),
        "n_beats": int(keep.sum()),
    }
    X_wide = extract_wide(x_filt, r_in[keep], delay, bc.half_window, WIDE_PAD).astype(np.float32)
    arrays = {"X": X, "X_wide": X_wide, "rr": rr[keep].astype(np.float32), "y": y_in[keep],
              "kind": kind_in[keep], "r_pos": r_in[keep],
              "rec_id": np.full(int(keep.sum()), rec.rec_id, dtype=np.int16)}
    return arrays, stats


# ---------------------------------------------------------------- cache

def cache_config(cfg: dict, det_cfg: dict, source: str) -> dict:
    """Full config that identifies a cache: beats config + detector config + version."""
    if source not in RR_SOURCES:
        raise ValueError(f"unknown rr_source {source!r}")
    return {"beats": {**cfg["beats"], "rr_source": source}, "detector": det_cfg,
            "beats_version": BEATS_VERSION}


def cache_dir(cfg_hash: str, source: str) -> Path:
    return CACHE_ROOT / f"beats_{source}_{cfg_hash}"


def save_record(out_dir: Path, rec_id: int, arrays: dict, stats: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / f"{rec_id}.npz", **arrays)
    (out_dir / f"{rec_id}.json").write_text(json.dumps(stats), encoding="utf-8")


def load_beats(out_dir: Path, records, trainable_only: bool = True) -> dict[str, np.ndarray]:
    """Concatenate cached records. trainable_only keeps matched non-Q beats (train/val)."""
    parts = [dict(np.load(out_dir / f"{r}.npz")) for r in records]
    data = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    if trainable_only:
        m = (data["kind"] == KIND_LABELED) & (data["y"] >= 0)
        data = {k: v[m] for k, v in data.items()}
    return data


def load_stats(out_dir: Path, records) -> dict[int, dict]:
    return {r: json.loads((out_dir / f"{r}.json").read_text(encoding="utf-8")) for r in records}


def assert_ds1_only(records) -> None:
    extra = set(records) - set(DS1)
    if extra:
        raise ValueError(f"beat cache builder is DS1-only here; got {sorted(extra)}")
