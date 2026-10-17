"""Causal 0.5-40 Hz band-pass (Butterworth, SOS) for R refinement and the CNN branch.

Offline: scipy.signal.sosfilt (zero initial state). Streaming: SosStream, a
per-sample Direct-Form-II-transposed implementation using the same operation
order as scipy's sosfilt, so both produce the same output.
"""

import numpy as np
from scipy import signal

FS = 360
BANDPASS_LOW_HZ = 0.5
BANDPASS_HIGH_HZ = 40.0
BANDPASS_ORDER = 2          # Butterworth prototype order -> 4 poles, 2 SOS sections
GROUP_DELAY_REF_HZ = 10.0


def design_bandpass(low: float = BANDPASS_LOW_HZ, high: float = BANDPASS_HIGH_HZ,
                    order: int = BANDPASS_ORDER, fs: float = FS) -> np.ndarray:
    return signal.butter(order, [low, high], btype="bandpass", fs=fs, output="sos")


def apply_offline(x: np.ndarray, sos: np.ndarray) -> np.ndarray:
    return signal.sosfilt(sos, np.asarray(x, dtype=np.float64))


def group_delay_samples(sos: np.ndarray, f_hz: float, fs: float = FS) -> float:
    """Group delay of the cascade at f_hz (sum over sections), in samples."""
    total = 0.0
    for sec in sos:
        _, gd = signal.group_delay((sec[:3], sec[3:]), w=[f_hz], fs=fs)
        total += float(gd[0])
    return total


def group_delay_compensation(sos: np.ndarray, fs: float = FS,
                             f_hz: float = GROUP_DELAY_REF_HZ) -> int:
    """Integer delay (samples) used to map filtered-signal indices back to raw time."""
    return int(round(group_delay_samples(sos, f_hz, fs)))


class SosStream:
    """Per-sample causal SOS filter with persistent state (DF2T, same op order as sosfilt)."""

    def __init__(self, sos: np.ndarray):
        sos = np.asarray(sos, dtype=np.float64)
        self._b0 = [float(s[0]) for s in sos]
        self._b1 = [float(s[1]) for s in sos]
        self._b2 = [float(s[2]) for s in sos]
        self._a1 = [float(s[4]) for s in sos]
        self._a2 = [float(s[5]) for s in sos]
        self._z0 = [0.0] * len(sos)
        self._z1 = [0.0] * len(sos)

    def push(self, x: float) -> float:
        x_cur = float(x)
        for s in range(len(self._b0)):
            x_new = self._b0[s] * x_cur + self._z0[s]
            self._z0[s] = self._b1[s] * x_cur - self._a1[s] * x_new + self._z1[s]
            self._z1[s] = self._b2[s] * x_cur - self._a2[s] * x_new
            x_cur = x_new
        return x_cur
