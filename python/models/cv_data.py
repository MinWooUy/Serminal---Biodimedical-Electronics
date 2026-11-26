"""Load the cached DS1 beat dataset (train/val beats only) for record-grouped CV."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from python.data.beats import cache_config, cache_dir, load_beats
from python.data.mitbih import REPO_ROOT, load_record
from python.data.splits import get_ds1_records, load_folds
from python.dsp.filters import group_delay_compensation
from python.dsp.pan_tompkins import detector_from_config
from python.experiment import config_hash, load_config
from python.models.training import assert_ds1


@dataclass
class CVData:
    data: dict[str, np.ndarray]        # X (n,360), rr (n,4), y, rec_id, r_pos, ...
    folds: list[list[int]]
    rec_len: dict[int, int]
    delay: int
    half: int
    fs: int
    cache: Path
    norm_eps: float
    norm_clip: float


def load_cv_data(beats_config: str | Path) -> CVData:
    bcfg = load_config(REPO_ROOT / beats_config)
    det_cfg = load_config(REPO_ROOT / bcfg["detector_config"])
    source = bcfg["beats"]["rr_source"]
    cdir = cache_dir(config_hash(cache_config(bcfg, det_cfg, source)), source)
    records = get_ds1_records()
    missing = [r for r in records if not (cdir / f"{r}.npz").exists()]
    if missing:
        raise FileNotFoundError(f"beat cache {cdir} incomplete ({missing}); run scripts/build_beats.py")
    data = load_beats(cdir, records)                  # matched, non-Q beats only
    assert_ds1(data["rec_id"])
    pt, sos = detector_from_config(det_cfg)
    rec_len = {r: len(load_record(r).signal) for r in records}
    norm = bcfg["beats"]["beat_norm"]
    return CVData(data, load_folds(), rec_len, group_delay_compensation(sos, pt.fs),
                  int(bcfg["beats"]["half_window"]), pt.fs, cdir, float(norm["eps"]), float(norm["clip"]))


def add_extra_train(data: dict[str, np.ndarray], extra: dict[str, np.ndarray]) -> tuple[dict, int, np.ndarray]:
    """Append training-only beats (e.g. SVDB) after the MIT-BIH beats.

    Returns (combined data, n_main, extra_idx). Indices < n_main are MIT-BIH (used for fold splits,
    inner-val and OOF); extra_idx (>= n_main) may ONLY be added to the inner-train indices.
    """
    keys = [k for k in data if k in extra]
    n_main = len(data["y"])
    combined = {k: np.concatenate([data[k], extra[k]]) for k in keys}
    return combined, n_main, np.arange(n_main, n_main + len(extra["y"]))


def with_extra_train(idx: dict[str, np.ndarray], extra_idx: np.ndarray, n_main: int) -> dict[str, np.ndarray]:
    """Fold indices with the extra beats added to inner-train only (asserts they never reach inner-val/outer)."""
    for k in ("outer", "inner_val", "buffer"):
        if len(idx[k]) and idx[k].max() >= n_main:
            raise ValueError(f"extra beats found in {k}")
    return idx | {"train": np.concatenate([idx["train"], extra_idx])}


def attach_extra_train(cfg: dict, data: dict[str, np.ndarray]) -> tuple[dict, int, np.ndarray, str | None]:
    """Load the config's optional training-only extra beats (cfg['extra_train'], A5: SVDB) and append them.
    Returns (data, n_main, extra_idx, cache name); without extra_train the data is returned unchanged."""
    n_main = len(data["y"])
    ex = cfg.get("extra_train")
    if not ex:
        return data, n_main, np.zeros(0, dtype=np.int64), None
    if ex.get("source") != "svdb":
        raise ValueError(f"unknown extra_train source {ex.get('source')!r}")
    from python.data.svdb import load_svdb_train, svdb_cache_dir, svdb_neg_cache_dir
    bcfg = load_config(REPO_ROOT / cfg["beats_config"])
    det = load_config(REPO_ROOT / bcfg["detector_config"])
    sv_dir = svdb_cache_dir(bcfg, det)
    neg = (svdb_neg_cache_dir(bcfg, det),) if ex.get("negative_fallback") else ()   # A5b
    sv = load_svdb_train(sv_dir, ex["variant"], int(ex.get("sample_seed", 0)), extra_caches=neg)
    combined, n_main, extra_idx = add_extra_train(data, sv)
    return combined, n_main, extra_idx, sv_dir.name + ("+" + neg[0].name if neg else "")
