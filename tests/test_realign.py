import numpy as np
import pytest

from python.dsp.realign import half_window_samples, pick_in_window, realign, realign_one


def _signal():
    x = np.zeros(400)
    x[100] = 1.0          # positive peak
    x[200] = -2.0         # negative peak (abs wins), small positive bump nearby
    x[205] = 0.5
    return x


def test_half_window():
    assert half_window_samples(360) == 18


def test_signed_and_abs():
    x = _signal()
    # filtered signal lags by delay=2: raw position 95 -> filtered centre 97
    assert realign_one(95, x, delay=2, half=18, mode="signed") == (98, False)
    assert realign_one(195, x, delay=2, half=18, mode="abs") == (198, False)
    assert realign_one(195, x, delay=2, half=18, mode="signed") == (203, False)


def test_none_mode_is_identity():
    assert realign_one(123, _signal(), delay=2, half=18, mode="none") == (123, False)


def test_edge_keeps_original():
    x = np.arange(400, dtype=float)          # monotonic: max always at the right edge
    pos, edge = realign_one(150, x, delay=2, half=18, mode="signed")
    assert (pos, edge) == (150, True)


def test_pick_first_occurrence():
    assert pick_in_window(np.array([0, 3, 3, 0]), "signed") == (1, False)


def test_vectorised_matches_scalar():
    x = np.random.default_rng(0).normal(size=1000)
    pos = np.array([30, 200, 500, 990])
    out, edge = realign(pos, x, 2, 18, "abs")
    for p, o, e in zip(pos, out, edge):
        assert (o, e) == realign_one(int(p), x, 2, 18, "abs")


def test_unknown_mode():
    with pytest.raises(ValueError):
        realign(np.array([10]), np.zeros(50), 2, 18, "max")
