import numpy as np

from python.dsp.evaluate_detector import (
    evaluate_record, match_beats, outside_vf, position_errors, tol_samples,
)


def test_tolerances():
    assert tol_samples(150, 360) == 54
    assert tol_samples(75, 360) == 27


def test_one_to_one():
    m = match_beats(np.array([1000]), np.array([990, 1005]), tol=54)
    assert m.tp == 1 and len(m.fn) == 0 and len(m.fp) == 1
    assert m.pairs.tolist() == [[0, 1]]         # closest detection wins


def test_window_boundary():
    m = match_beats(np.array([1000, 2000]), np.array([1054, 2055]), tol=54)
    assert m.pairs.tolist() == [[0, 0]]
    assert m.fn.tolist() == [1] and m.fp.tolist() == [1]


def test_greedy_global_not_sequential():
    # ref 0 could take det 1 (dist 40) but det 1 is closer to ref 1 (dist 5)
    m = match_beats(np.array([100, 145]), np.array([60, 140]), tol=54)
    assert m.pairs.tolist() == [[0, 0], [1, 1]]


def test_outside_vf():
    vf = np.array([[100, 200]])
    assert outside_vf(np.array([99, 100, 150, 200, 201]), vf).tolist() == [True, False, False, False, True]


def test_evaluate_excludes_vf_and_counts_q():
    ann = np.array([50, 150, 300])          # 150 inside VF; 300 could be a Q beat: still counted
    det = np.array([52, 160, 400])
    ev = evaluate_record(ann, det, det, ann, np.array([[120, 180]]), fs=360)
    assert ev["150ms"] == {"tp": 1, "fn": 1, "fp": 1, "se": 0.5, "ppv": 0.5}
    assert ev["n_ref"] == 2 and ev["n_det"] == 2


def test_position_errors_ms():
    e = position_errors(np.array([0, 100]), np.array([3, 97]), np.array([[0, 0], [1, 1]]), fs=360)
    assert e["n"] == 2
    np.testing.assert_allclose(e["mean_ms"], 0.0)
    np.testing.assert_allclose(e["std_ms"], 3 * 1000 / 360)
