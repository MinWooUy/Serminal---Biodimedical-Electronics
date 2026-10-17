import numpy as np

from python.viz.common import FP, Q, VF
from scripts.beat_gallery import ROW_FP, gallery_cells, rr_ratio


def _labels(seed=0, n=3000):
    rng = np.random.default_rng(seed)
    true = rng.choice([0, 1, 2, 3, FP, Q, VF], size=n, p=[0.6, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05])
    pred = rng.integers(0, 4, size=n)
    return true, pred


def test_cells_partition_and_counts():
    true, pred = _labels()
    cells = gallery_cells(true, pred, 16, seed=0)
    for r in range(4):
        for c in range(4):
            assert len(cells[(r, c)]["idx"]) == np.sum((true == r) & (pred == c))
    for c in range(4):
        assert len(cells[(ROW_FP, c)]["idx"]) == np.sum((true == FP) & (pred == c))
    members = np.concatenate([v["idx"] for v in cells.values()])
    assert len(members) == len(np.unique(members)) == np.sum(true >= 0) + np.sum(true == FP)   # Q / VF skipped


def test_sample_bounded_subset_and_reproducible():
    true, pred = _labels()
    a = gallery_cells(true, pred, 16, seed=0)
    b = gallery_cells(true, pred, 16, seed=0)
    c = gallery_cells(true, pred, 16, seed=1)
    for k in a:
        assert len(a[k]["sample"]) == min(16, len(a[k]["idx"]))
        assert np.isin(a[k]["sample"], a[k]["idx"]).all()
        assert np.array_equal(a[k]["sample"], b[k]["sample"])
    assert any(not np.array_equal(a[k]["sample"], c[k]["sample"]) for k in a if len(a[k]["idx"]) > 16)


def test_rr_ratio():
    assert np.allclose(rr_ratio(np.array([[360, 400, 8 * 450, 8], [300, 300, 600, 2]])), [0.8, 1.0])
