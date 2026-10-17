"""Only python/data/mitbih.py (MIT-BIH, DS2 lock) and python/data/svdb.py (SVDB, training-only extra data)
may read records/annotations, so the DS2 lock cannot be bypassed."""

import re

from tests.conftest import REPO_ROOT

ALLOWED = REPO_ROOT / "python" / "data" / "mitbih.py"
ALLOWED_EXTRA = {(REPO_ROOT / "python" / "data" / "svdb.py").resolve()}
PATTERN = re.compile(r"\b(rdrecord|rdann|rdsamp)\b")


def test_only_mitbih_reads_wfdb():
    offenders = []
    for sub in ("python", "scripts"):
        for path in (REPO_ROOT / sub).rglob("*.py"):
            if path.resolve() == ALLOWED.resolve() or path.resolve() in ALLOWED_EXTRA:
                continue
            for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if PATTERN.search(line):
                    offenders.append(f"{path.relative_to(REPO_ROOT)}:{n}: {line.strip()}")
    assert not offenders, "wfdb readers outside mitbih.py:\n" + "\n".join(offenders)


def test_mitbih_does_read_wfdb():
    # Guard against the pattern silently matching nothing.
    assert PATTERN.search(ALLOWED.read_text(encoding="utf-8"))
