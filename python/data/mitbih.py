"""MIT-BIH Arrhythmia loader.

This is the ONLY module allowed to call wfdb.rdrecord / wfdb.rdann
(enforced by tests/test_wfdb_isolation.py), so the DS2 lock cannot be bypassed.
"""

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import wfdb

from python.data.constants import (
    AAMI_MAP, ALL_RECORDS, CLASSES, DS1_EXPECTED_COUNTS, DS2, FS_MITBIH,
    NON_BEAT_SYMBOLS, PACED, Q_LABEL, SANITY_TOLERANCE, VF_OFFSET, VF_ONSET,
    require_ds2_access,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = REPO_ROOT / "data" / "raw" / "mitdb"  # anchored: independent of cwd
LEAD = "MLII"


@dataclass(frozen=True)
class Record:
    rec_id: int
    fs: int
    signal: np.ndarray        # MLII, physical units (mV), float64, shape (n,)
    r_samples: np.ndarray     # beat annotation positions (not re-aligned), int64
    symbols: np.ndarray       # original beat symbols, str
    aami: np.ndarray          # AAMI class 0..3 (N,S,V,F), Q = -1, int8
    vf_segments: np.ndarray   # ventricular flutter [start, end] in samples, int64, shape (m, 2)
    adc: np.ndarray           # MLII raw ADC counts, int16; signal == (adc - adc_zero) / adc_gain
    adc_zero: int
    adc_gain: float


@dataclass(frozen=True)
class ParsedAnnotations:
    r_samples: np.ndarray
    symbols: np.ndarray
    aami: np.ndarray
    vf_segments: np.ndarray


def parse_annotations(samples: Iterable[int], symbols: Iterable[str],
                      sig_len: int) -> ParsedAnnotations:
    """Split raw annotations into beats (all kept, incl. Q) and VF segments.

    Unknown symbols raise ValueError instead of being silently dropped.
    An unclosed '[' ends at the last sample; a ']' without '[' starts at 0.
    """
    samples = np.asarray(list(samples), dtype=np.int64)
    symbols = list(symbols)
    if len(samples) != len(symbols):
        raise ValueError("samples and symbols length mismatch")

    beat_idx = []
    vf_segments = []
    vf_start = None
    for i, (s, sym) in enumerate(zip(samples, symbols)):
        if sym in AAMI_MAP:
            beat_idx.append(i)
        elif sym == VF_ONSET:
            if vf_start is not None:
                raise ValueError(f"nested '[' at sample {s}")
            vf_start = int(s)
        elif sym == VF_OFFSET:
            vf_segments.append((0 if vf_start is None else vf_start, int(s)))
            vf_start = None
        elif sym not in NON_BEAT_SYMBOLS:
            raise ValueError(f"unknown annotation symbol {sym!r} at sample {s}")
    if vf_start is not None:
        vf_segments.append((vf_start, sig_len - 1))

    beat_idx = np.asarray(beat_idx, dtype=np.int64)
    r_samples = samples[beat_idx]
    if np.any((r_samples < 0) | (r_samples >= sig_len)):
        raise ValueError("beat annotation outside signal range")
    beat_symbols = np.asarray([symbols[i] for i in beat_idx], dtype=str)
    aami = np.asarray([AAMI_MAP[s] for s in beat_symbols], dtype=np.int8)
    vf = np.asarray(vf_segments, dtype=np.int64).reshape(-1, 2)
    return ParsedAnnotations(r_samples, beat_symbols, aami, vf)


def _check_record_access(rec_id: int) -> None:
    # Called before touching any file.
    if rec_id in PACED:
        raise ValueError(f"record {rec_id} is paced and excluded")
    if rec_id not in ALL_RECORDS:
        raise ValueError(f"record {rec_id} is not in DS1/DS2")
    if rec_id in DS2:
        require_ds2_access()


def load_record(rec_id: int, raw_dir: str | Path = DEFAULT_RAW_DIR) -> Record:
    _check_record_access(rec_id)
    path = str(Path(raw_dir) / str(rec_id))

    rec = wfdb.rdrecord(path, physical=False)
    if rec.fs != FS_MITBIH:
        raise ValueError(f"record {rec_id}: fs={rec.fs}, expected {FS_MITBIH}")
    if LEAD not in rec.sig_name:
        raise ValueError(f"record {rec_id}: no {LEAD} lead (leads={rec.sig_name})")
    ch = rec.sig_name.index(LEAD)
    adc_zero, baseline = int(rec.adc_zero[ch]), int(rec.baseline[ch])
    if adc_zero != baseline:
        raise ValueError(f"record {rec_id}: adc_zero {adc_zero} != baseline {baseline}")
    adc_gain = float(rec.adc_gain[ch])
    adc = rec.d_signal[:, ch].astype(np.int16)
    # Same expression as the streaming detector uses, so both see identical floats.
    signal = (adc.astype(np.int64) - adc_zero) / adc_gain

    ann = wfdb.rdann(path, "atr")
    parsed = parse_annotations(ann.sample, ann.symbol, len(signal))
    return Record(
        rec_id=rec_id,
        fs=int(rec.fs),
        signal=signal,
        r_samples=parsed.r_samples,
        symbols=parsed.symbols,
        aami=parsed.aami,
        vf_segments=parsed.vf_segments,
        adc=adc,
        adc_zero=adc_zero,
        adc_gain=adc_gain,
    )


def count_labels(records: Iterable[Record]) -> dict[str, int]:
    """Count beats per AAMI class, Q included."""
    names = {i: c for i, c in enumerate(CLASSES)} | {Q_LABEL: "Q"}
    counts = Counter()
    for r in records:
        counts.update(int(a) for a in r.aami)
    return {names[k]: counts.get(k, 0) for k in names}


def check_ds1_sanity(counts: dict[str, int],
                     tol: float = SANITY_TOLERANCE) -> dict[str, float]:
    """Return relative deviation per class; raise if any exceeds tol."""
    dev = {c: (counts[c] - e) / e for c, e in DS1_EXPECTED_COUNTS.items()}
    bad = {c: d for c, d in dev.items() if abs(d) > tol}
    if bad:
        raise ValueError(f"DS1 counts deviate beyond {tol:.0%}: {bad} (counts={counts})")
    return dev


def download(dl_dir: str | Path = DEFAULT_RAW_DIR) -> None:
    """Download all non-paced MIT-BIH records (DS1 + DS2 files; no labels are read)."""
    dl_dir = Path(dl_dir)
    missing = [str(r) for r in ALL_RECORDS
               if not all((dl_dir / f"{r}.{ext}").exists() for ext in ("hea", "dat", "atr"))]
    if missing:
        wfdb.dl_database("mitdb", dl_dir=str(dl_dir), records=missing, annotators="atr")
