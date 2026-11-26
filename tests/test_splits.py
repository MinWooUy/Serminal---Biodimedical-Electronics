import json

import pytest

from python.data.constants import DS1
from python.data.mitbih import count_labels
from python.data.splits import (
    DEFAULT_FOLDS_PATH, fold_counts, load_folds, make_group_kfold, save_folds,
    train_val_records,
)
from tests.conftest import REPO_ROOT


def test_default_paths_anchored_to_repo():
    # Running scripts from another cwd must not create stray data/config copies.
    from python.data.mitbih import DEFAULT_RAW_DIR
    for p in (DEFAULT_RAW_DIR, DEFAULT_FOLDS_PATH):
        assert p.is_absolute()
        assert p.resolve().is_relative_to(REPO_ROOT)


def _synthetic_counts():
    # 22 fake records; S and F present in only a few, one record dominates F.
    counts = {}
    for i, r in enumerate(DS1):
        counts[r] = {"N": 2000 + 10 * i, "S": 0, "V": 50 * (i % 4), "F": 0, "Q": 0}
    for r, s in zip(DS1[:7], (300, 100, 90, 80, 70, 60, 5)):
        counts[r]["S"] = s
    for r, f in zip(DS1[7:13], (370, 15, 10, 5, 3, 2)):
        counts[r]["F"] = f
    return counts


def test_folds_partition_and_classes():
    counts = _synthetic_counts()
    folds = make_group_kfold(counts, k=5, seed=42, n_trials=2000)
    flat = [r for f in folds for r in f]
    assert sorted(flat) == sorted(DS1) and len(flat) == len(set(flat))
    assert sorted(len(f) for f in folds) == [4, 4, 4, 5, 5]
    for c in fold_counts(folds, counts):
        assert c["S"] > 0 and c["F"] > 0


def test_folds_deterministic():
    counts = _synthetic_counts()
    a = make_group_kfold(counts, k=5, seed=42, n_trials=2000)
    b = make_group_kfold(counts, k=5, seed=42, n_trials=2000)
    assert a == b


def test_folds_impossible_raises():
    counts = _synthetic_counts()
    for r in DS1[9:13]:
        counts[r]["F"] = 0  # only 2 records with F < k
    with pytest.raises(ValueError, match="class F"):
        make_group_kfold(counts, k=5)


def test_train_val_disjoint():
    folds = make_group_kfold(_synthetic_counts(), k=5, seed=0, n_trials=500)
    for i in range(5):
        train, val = train_val_records(folds, i)
        assert not set(train) & set(val)
        assert sorted(train + val) == sorted(DS1)


def test_save_load_roundtrip(tmp_path):
    counts = _synthetic_counts()
    folds = make_group_kfold(counts, k=5, seed=1, n_trials=500)
    path = tmp_path / "f.json"
    save_folds(folds, counts, path, seed=1, n_trials=500)
    assert load_folds(path) == folds
    data = json.loads(path.read_text())
    assert all(set(f["counts"]) == {"N", "S", "V", "F", "Q"} for f in data["folds"])


@pytest.mark.needs_data
def test_real_folds(ds1_records):
    path = DEFAULT_FOLDS_PATH
    if not path.exists():
        pytest.skip("run scripts/make_folds.py first")
    folds = load_folds(path)
    stored = json.loads(path.read_text())["folds"]
    for i, recs in enumerate(folds):
        c = count_labels(ds1_records[r] for r in recs)
        print(f"fold {i}: {recs} {c}")
        assert c["S"] > 0 and c["F"] > 0
        assert c == stored[i]["counts"]  # JSON matches the data
