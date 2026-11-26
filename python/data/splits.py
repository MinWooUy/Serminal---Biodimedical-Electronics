"""DS1/DS2 record lists and record-grouped k-fold on DS1.

Folds are found by a seeded random search over record assignments:
every fold must contain S and F beats; among valid assignments, the one with
the most balanced per-class beat counts across folds is kept.
"""

import json
from pathlib import Path

import numpy as np

from python.data.constants import CLASSES, DS1, DS2

DEFAULT_FOLDS_PATH = Path(__file__).resolve().parents[2] / "configs" / "splits" / "ds1_k5.json"
REQUIRED_CLASSES = ("S", "F")


def get_ds1_records() -> list[int]:
    return list(DS1)


def get_ds2_records() -> list[int]:
    """Record IDs only. Labels are locked in mitbih.load_record."""
    return list(DS2)


def fold_counts(folds: list[list[int]],
                per_record_counts: dict[int, dict[str, int]]) -> list[dict[str, int]]:
    out = []
    for recs in folds:
        keys = per_record_counts[recs[0]].keys()
        out.append({c: int(sum(per_record_counts[r][c] for r in recs)) for c in keys})
    return out


def _imbalance(counts: np.ndarray) -> float:
    # counts: (k, n_classes). Sum of coefficient of variation over classes.
    mean = counts.mean(axis=0)
    return float(np.sum(counts.std(axis=0) / np.maximum(mean, 1e-9)))


def make_group_kfold(per_record_counts: dict[int, dict[str, int]], k: int = 5,
                     seed: int = 42, n_trials: int = 20000) -> list[list[int]]:
    records = sorted(per_record_counts)
    if len(records) < k:
        raise ValueError(f"need at least {k} records, got {len(records)}")
    for c in REQUIRED_CLASSES:
        n_with = sum(per_record_counts[r][c] > 0 for r in records)
        if n_with < k:
            raise ValueError(f"only {n_with} records contain class {c}; cannot fill {k} folds")

    mat = np.array([[per_record_counts[r][c] for c in CLASSES] for r in records])
    req_cols = [CLASSES.index(c) for c in REQUIRED_CLASSES]
    rng = np.random.default_rng(seed)

    best, best_score = None, np.inf
    for _ in range(n_trials):
        perm = rng.permutation(len(records))
        groups = np.array_split(perm, k)
        counts = np.stack([mat[g].sum(axis=0) for g in groups])
        if np.any(counts[:, req_cols] == 0):
            continue
        score = _imbalance(counts)
        if score < best_score:
            best_score, best = score, groups

    if best is None:
        raise RuntimeError(f"no valid assignment found in {n_trials} trials")
    folds = [sorted(records[i] for i in g) for g in best]
    return sorted(folds)


def train_val_records(folds: list[list[int]], fold: int) -> tuple[list[int], list[int]]:
    val = list(folds[fold])
    train = sorted(r for i, f in enumerate(folds) if i != fold for r in f)
    return train, val


def save_folds(folds: list[list[int]], per_record_counts: dict[int, dict[str, int]],
               path: str | Path = DEFAULT_FOLDS_PATH, *, seed: int, n_trials: int) -> None:
    counts = fold_counts(folds, per_record_counts)
    totals = {c: sum(fc[c] for fc in counts) for c in counts[0]}
    data = {
        "k": len(folds),
        "seed": seed,
        "n_trials": n_trials,
        "method": "seeded random search; each fold has S and F; "
                  "minimize sum of per-class CV of fold counts",
        "folds": [{"fold": i, "records": f, "counts": c}
                  for i, (f, c) in enumerate(zip(folds, counts))],
        "totals": totals,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def load_folds(path: str | Path = DEFAULT_FOLDS_PATH) -> list[list[int]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    folds = [f["records"] for f in sorted(data["folds"], key=lambda f: f["fold"])]
    flat = [r for f in folds for r in f]
    if sorted(flat) != sorted(DS1) or len(flat) != len(set(flat)):
        raise ValueError("folds file does not partition DS1 exactly")
    return folds
