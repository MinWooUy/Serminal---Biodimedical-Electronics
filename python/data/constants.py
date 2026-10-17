"""Dataset constants for MIT-BIH Arrhythmia (record lists, AAMI mapping, DS2 lock)."""

import os
from collections.abc import Mapping

FS_MITBIH = 360

PACED = (102, 104, 107, 217)

DS1 = (101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122,
       124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230)

DS2 = (100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210,
       212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234)

ALL_RECORDS = tuple(sorted(DS1 + DS2))

CLASSES = ("N", "S", "V", "F")
Q_LABEL = -1

# Beat annotation symbol -> AAMI class index (0..3), Q -> -1.
AAMI_MAP = {
    "N": 0, "L": 0, "R": 0, "e": 0, "j": 0,
    "A": 1, "a": 1, "J": 1, "S": 1,
    "V": 2, "E": 2,
    "F": 3,
    "/": Q_LABEL, "f": Q_LABEL, "Q": Q_LABEL,
}

# Non-beat annotation symbols (PhysioBank annotation codes). '[' and ']' mark
# ventricular flutter/fibrillation onset/offset; '!' is a flutter wave (non-beat).
VF_ONSET = "["
VF_OFFSET = "]"
NON_BEAT_SYMBOLS = frozenset({
    "[", "]", "!", "x", "(", ")", "p", "t", "u", "`", "'", "^",
    "|", "~", "+", "s", "T", "*", "D", "=", '"', "@",
})

# Expected DS1 beat counts per AAMI class (from annotations), used as a sanity gate.
DS1_EXPECTED_COUNTS = {"N": 45800, "S": 944, "V": 3790, "F": 414}
SANITY_TOLERANCE = 0.02

DS2_ENV_VAR = "ECG_ALLOW_DS2"


class DS2LockedError(PermissionError):
    """Raised when DS2 labels are accessed without explicit user permission."""


def require_ds2_access(env: Mapping[str, str] | None = None) -> None:
    """Raise DS2LockedError unless ECG_ALLOW_DS2 is exactly "1"."""
    env = os.environ if env is None else env
    if env.get(DS2_ENV_VAR) != "1":
        raise DS2LockedError(
            f"DS2 is locked. Only the user may run scripts/final_eval_ds2.py "
            f"with {DS2_ENV_VAR}=1."
        )
