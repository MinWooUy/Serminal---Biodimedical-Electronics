"""MIT-BIH Supraventricular Arrhythmia Database (SVDB) loader — TRAINING-ONLY extra data (A5).

Conventions (CLAUDE.md):
  - 128 Hz -> 360 Hz with resample_poly(x, 45, 16); annotation sample x 2.8125, rounded;
  - mV x 200, rounded to integers -> fed to the integer Pan-Tompkins (adc_zero = 0, gain 200);
  - channel chosen per record (option A, strict): among channels whose QRS is mostly positive
    (> 50% of annotated beats, sign of the 0.5-40 Hz signal at the abs-re-aligned R), the one with
    the higher detector Se (tie -> higher +P). A record is KEPT only if such a positive channel exists
    AND it has Se and +P >= 97%; otherwise the whole record is excluded (no fallback to a negative
    channel: keeps an MLII-like morphology and avoids N beats with negative QRS blurring N/V);
  - same AAMI table as MIT-BIH; unknown symbols raise (parse_annotations), except the SVDB-only
    remap below: 'B' (bundle branch block beat, unspecified; 1 beat in record 868) is treated as Q
    (kept in the RR sequence, dropped from the dataset after RR).
Besides python/data/mitbih.py, this is the only module allowed to call wfdb readers.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import wfdb
from scipy.signal import resample_poly

from python.data.mitbih import REPO_ROOT, Record, parse_annotations
from python.dsp.evaluate_detector import evaluate_record
from python.dsp.pan_tompkins import PTConfig, detect_offline, frontend_offline, stage_max_abs
from python.dsp.realign import half_window_samples, realign

DEFAULT_SVDB_DIR = REPO_ROOT / "data" / "raw" / "svdb"
SVDB_FS = 128
UP, DOWN = 45, 16                    # 128 * 45 / 16 = 360
ANN_SCALE = UP / DOWN                # 2.8125
GAIN = 200.0
MIN_SE_PPV = 0.97
SVDB_SYMBOL_REMAP = {"B": "Q"}       # user decision: unspecified BBB beat -> Q (SVDB only)


def available_records(raw_dir: str | Path = DEFAULT_SVDB_DIR) -> list[int]:
    raw_dir = Path(raw_dir)
    ids = []
    for h in raw_dir.glob("*.hea"):
        if (raw_dir / f"{h.stem}.dat").exists() and (raw_dir / f"{h.stem}.atr").exists() and h.stem.isdigit():
            ids.append(int(h.stem))
    return sorted(ids)


def to_360(x_mv: np.ndarray) -> np.ndarray:
    return resample_poly(np.asarray(x_mv, dtype=np.float64), UP, DOWN)


def scale_ann(samples: np.ndarray) -> np.ndarray:
    return np.round(np.asarray(samples, dtype=np.float64) * ANN_SCALE).astype(np.int64)


def drop_out_of_range(samples, symbols, sig_len: int) -> tuple[np.ndarray, list[str]]:
    """SVDB-only data fix: drop annotations at or after the last sample (record 860 has an 'N'
    at sample 230400 == sig_len). Such a beat is the record's last one and would be dropped anyway
    (no rr_next). parse_annotations keeps raising for MIT-BIH."""
    samples = np.asarray(samples, dtype=np.int64)
    keep = samples < sig_len
    return samples[keep], [s for s, k in zip(symbols, keep) if k]


def load_channel(rec_id: int, ch: int, raw_dir: str | Path = DEFAULT_SVDB_DIR) -> Record:
    """One SVDB channel as a 360 Hz MIT-BIH-like Record (signal == adc / 200)."""
    path = str(Path(raw_dir) / str(rec_id))
    rec = wfdb.rdrecord(path, channels=[ch])
    if rec.fs != SVDB_FS:
        raise ValueError(f"SVDB {rec_id}: fs={rec.fs}")
    x = to_360(rec.p_signal[:, 0])
    adc = np.round(x * GAIN).astype(np.int64)
    if np.abs(adc).max() >= 2**15:
        raise ValueError(f"SVDB {rec_id} ch {ch}: amplitude out of int16 range")
    ann = wfdb.rdann(path, "atr")
    samples, symbols = drop_out_of_range(ann.sample, ann.symbol, rec.sig_len)
    symbols = [SVDB_SYMBOL_REMAP.get(s, s) for s in symbols]
    parsed = parse_annotations(scale_ann(samples), symbols, len(adc))
    return Record(rec_id=rec_id, fs=360, signal=adc / GAIN, r_samples=parsed.r_samples,
                  symbols=parsed.symbols, aami=parsed.aami, vf_segments=parsed.vf_segments,
                  adc=adc.astype(np.int16), adc_zero=0, adc_gain=GAIN)


def n_channels(rec_id: int, raw_dir: str | Path = DEFAULT_SVDB_DIR) -> int:
    return wfdb.rdheader(str(Path(raw_dir) / str(rec_id))).n_sig


@dataclass
class ChannelEval:
    ch: int
    se: float
    ppv: float
    tp: int
    fn: int
    fp: int
    pos_frac: float
    max_mwi_log2: float


def evaluate_channel(rec: Record, pt: PTConfig, sos: np.ndarray, ch: int) -> ChannelEval:
    res = detect_offline(rec.adc, rec.adc_zero, rec.adc_gain, pt, sos)
    half = half_window_samples(pt.fs)
    ann_abs, _ = realign(rec.r_samples, res.x_filt, res.delay, half, "abs")
    ann_re, _ = realign(rec.r_samples, res.x_filt, res.delay, half, pt.realign_mode)
    ev = evaluate_record(rec.r_samples, res.r_refined, res.r_coarse, ann_re, rec.vf_segments, pt.fs)
    e = ev["150ms"]
    idx = np.clip(ann_abs + res.delay, 0, len(res.x_filt) - 1)
    pos = float(np.mean(res.x_filt[idx] > 0)) if len(idx) else 0.0
    mwi = stage_max_abs(frontend_offline(rec.adc, rec.adc_zero, pt))["mwi"]
    return ChannelEval(ch, e["se"], e["ppv"], e["tp"], e["fn"], e["fp"], pos, float(np.log2(max(mwi, 1))))


def choose_channel(evals: list[ChannelEval]) -> tuple[ChannelEval | None, str]:
    """(chosen channel or None, reason). Rule in the module docstring (option A, strict)."""
    positive = [e for e in evals if e.pos_frac > 0.5]
    if not positive:
        return None, "no mostly-positive channel"
    best = max(positive, key=lambda e: (e.se, e.ppv))
    if best.se < MIN_SE_PPV or best.ppv < MIN_SE_PPV:
        return None, f"positive channel {best.ch} below {MIN_SE_PPV:.0%} (Se {best.se:.4f}, +P {best.ppv:.4f})"
    return best, "ok"


# ---------------------------------------------------------------- beat cache (training-only)

SVDB_RULES = {"version": "svdb-v1", "channel_rule": "A-strict", "min_se_ppv": MIN_SE_PPV,
              "resample": [UP, DOWN], "gain": GAIN, "symbol_remap": SVDB_SYMBOL_REMAP,
              "drop_ann_out_of_range": True}


def svdb_cache_dir(beats_cfg: dict, det_cfg: dict) -> Path:
    from python.data.beats import CACHE_ROOT, cache_config
    from python.experiment import config_hash
    return CACHE_ROOT / f"svdb_beats_{config_hash(cache_config(beats_cfg, det_cfg, 'detector') | {'svdb': SVDB_RULES})}"


def build_svdb_record(rec_id: int, pt: PTConfig, sos: np.ndarray, bc, raw_dir: str | Path = DEFAULT_SVDB_DIR):
    """Channel choice (rule A) + the unchanged MIT-BIH beat pipeline (beats.build_record) on the chosen channel.
    Returns (arrays or None if the record is excluded, stats)."""
    from python.data.beats import build_record
    recs = [load_channel(rec_id, ch, raw_dir) for ch in range(n_channels(rec_id, raw_dir))]
    evals = [evaluate_channel(r, pt, sos, ch) for ch, r in enumerate(recs)]
    best, reason = choose_channel(evals)
    stats = {"rec": rec_id, "reason": reason, "chosen": best.ch if best else None,
             "channels": [vars(e) for e in evals],
             "n_ann": {c: int(np.sum(recs[0].aami == k)) for k, c in ((0, "N"), (1, "S"), (2, "V"), (3, "F"), (-1, "Q"))}}
    if best is None:
        return None, stats
    arrays, bstats = build_record(recs[best.ch], pt, sos, bc)
    return arrays, stats | bstats


def load_svdb_train(cache: Path, variant: str = "all", seed: int = 0, extra_caches: tuple = ()) -> dict[str, np.ndarray]:
    """Matched, non-Q SVDB beats of the kept records. variant 'all' or 'n25' (all S/V/F + 25% of N,
    sampled once with a fixed seed, identical for every fold and training seed).
    extra_caches (A5b): further caches (e.g. negative-channel fallback records), each sampled
    SEPARATELY with the same rule and seed and appended, so the main cache gives exactly the same
    beats as without them."""
    parts = [_load_one(cache, variant, seed)] + [_load_one(c, variant, seed) for c in extra_caches]
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def _load_one(cache: Path, variant: str, seed: int) -> dict[str, np.ndarray]:
    from python.data.beats import load_beats
    import json
    manifest = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
    data = load_beats(cache, manifest["kept"])
    if variant.startswith("n") and variant[1:].isdigit():          # nXX: all S/V/F + XX% of N
        frac = int(variant[1:]) / 100.0
        if not 0 < frac <= 1:
            raise ValueError(f"bad SVDB variant {variant!r}")
        rng = np.random.default_rng(seed)
        n_idx = np.flatnonzero(data["y"] == 0)
        keep_n = rng.choice(n_idx, size=int(round(frac * len(n_idx))), replace=False)
        keep = np.sort(np.concatenate([np.flatnonzero(data["y"] != 0), keep_n]))
        data = {k: v[keep] for k, v in data.items()}
    elif variant != "all":
        raise ValueError(f"unknown SVDB variant {variant!r}")
    return data


# ---------------------------------------------------------------- A5b: negative-channel fallback

def choose_negative_fallback(evals: list[ChannelEval]) -> tuple[ChannelEval | None, str]:
    """A5b only, for records EXCLUDED by rule A: the mostly-negative channel (pos_frac <= 0.5) with the
    highest Se (tie -> +P) among those with Se and +P >= 97%; None if no such channel."""
    neg = [e for e in evals if e.pos_frac <= 0.5 and e.se >= MIN_SE_PPV and e.ppv >= MIN_SE_PPV]
    if not neg:
        return None, "no negative channel >= 97%"
    return max(neg, key=lambda e: (e.se, e.ppv)), "negative-channel fallback"


SVDB_RULES_NEG = SVDB_RULES | {"version": "svdb-neg-v1", "channel_rule": "A5b-negative-fallback",
                               "records": "only records excluded by rule A"}


def svdb_neg_cache_dir(beats_cfg: dict, det_cfg: dict) -> Path:
    from python.data.beats import CACHE_ROOT, cache_config
    from python.experiment import config_hash
    h = config_hash(cache_config(beats_cfg, det_cfg, "detector") | {"svdb": SVDB_RULES_NEG})
    return CACHE_ROOT / f"svdb_neg_beats_{h}"


def build_svdb_negative_record(rec_id: int, pt: PTConfig, sos: np.ndarray, bc,
                               raw_dir: str | Path = DEFAULT_SVDB_DIR):
    """A5b: same pipeline as build_svdb_record, but on the negative channel chosen by choose_negative_fallback.
    Must only be called for records excluded by rule A."""
    from python.data.beats import build_record
    recs = [load_channel(rec_id, ch, raw_dir) for ch in range(n_channels(rec_id, raw_dir))]
    evals = [evaluate_channel(r, pt, sos, ch) for ch, r in enumerate(recs)]
    best_a, _ = choose_channel(evals)
    if best_a is not None:
        raise ValueError(f"SVDB {rec_id} is kept by rule A; the negative fallback applies only to excluded records")
    best, reason = choose_negative_fallback(evals)
    stats = {"rec": rec_id, "reason": reason, "chosen": best.ch if best else None,
             "channels": [vars(e) for e in evals],
             "n_ann": {c: int(np.sum(recs[0].aami == k)) for k, c in ((0, "N"), (1, "S"), (2, "V"), (3, "F"), (-1, "Q"))}}
    if best is None:
        return None, stats
    arrays, bstats = build_record(recs[best.ch], pt, sos, bc)
    return arrays, stats | bstats
