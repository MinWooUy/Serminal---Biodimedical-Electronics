"""Shared R re-alignment: local maximum within +/-0.05 s on the 0.5-40 Hz filtered signal.

Used for BOTH annotations and detections (offline and streaming) so beat
windows are centred by the same rule.

Positions are in raw-signal time. The filtered signal lags by `delay` samples
(rounded group delay), so the search window in filtered indices is
[pos + delay - half, pos + delay + half]; the result is mapped back by -delay.
If the maximum falls on the window boundary there is no true local maximum in
the window: the original position is kept and an edge hit is reported.
"""

import numpy as np

MODES = ("signed", "abs", "none")
HALF_WINDOW_S = 0.05


def half_window_samples(fs: int) -> int:
    return int(round(HALF_WINDOW_S * fs))


def pick_in_window(window: np.ndarray, mode: str) -> tuple[int, bool]:
    """Return (index of the maximum in window, edge_hit). First occurrence wins."""
    if mode == "signed":
        k = int(np.argmax(window))
    elif mode == "abs":
        k = int(np.argmax(np.abs(window)))
    else:
        raise ValueError(f"unknown realign mode {mode!r}")
    return k, k == 0 or k == len(window) - 1


def realign_one(pos: int, x_filt: np.ndarray, delay: int, half: int, mode: str,
                offset: int = 0) -> tuple[int, bool]:
    """Re-align one position. x_filt[i] holds filtered sample (offset + i)."""
    if mode == "none":
        return int(pos), False
    c = pos + delay
    lo = max(c - half, offset)
    hi = min(c + half, offset + len(x_filt) - 1)
    if hi <= lo:
        return int(pos), True
    k, edge = pick_in_window(x_filt[lo - offset:hi - offset + 1], mode)
    if edge:
        return int(pos), True
    return int(lo + k - delay), False


def realign(positions: np.ndarray, x_filt: np.ndarray, delay: int, half: int,
            mode: str) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised over positions; returns (new_positions int64, edge_hit bool)."""
    if mode not in MODES:
        raise ValueError(f"unknown realign mode {mode!r}")
    out = np.empty(len(positions), dtype=np.int64)
    edge = np.zeros(len(positions), dtype=bool)
    for i, p in enumerate(positions):
        out[i], edge[i] = realign_one(int(p), x_filt, delay, half, mode)
    return out, edge
