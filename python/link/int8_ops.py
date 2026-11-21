"""The three integer operations of python/quant/int8.py the tile host needs, without torch (for the stand-alone hand-off package).

Copied, not changed: tests/test_handoff_tile.py checks that they equal int8.Q, int8.rshift_round and int8._maxpool2.
"""

import numpy as np

Q = 127


def rshift_round(acc: np.ndarray, s) -> np.ndarray:
    """(acc + (1 << (s - 1))) >> s on int64 with arithmetic shift; per-channel shifts broadcast over axis 1."""
    acc = np.asarray(acc, dtype=np.int64)
    s = np.asarray(s, dtype=np.int64)
    if s.ndim == 0:
        s = int(s)
        if s > 0:
            return (acc + (np.int64(1) << np.int64(s - 1))) >> np.int64(s)
        return acc if s == 0 else acc << np.int64(-s)
    out = np.empty_like(acc)
    for c, sc in enumerate(s.tolist()):
        out[:, c] = rshift_round(acc[:, c], sc)
    return out


def maxpool2(x: np.ndarray) -> np.ndarray:
    """MaxPool(2) over the last axis of (B, C, L)."""
    L = x.shape[2] // 2 * 2
    return np.maximum(x[:, :, 0:L:2], x[:, :, 1:L:2])
