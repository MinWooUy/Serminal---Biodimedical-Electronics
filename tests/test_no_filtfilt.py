"""No zero-phase (non-causal) filtering anywhere in the code (CLAUDE.md rule 3).

Scans code/config files (*.py, *.yaml) in the repo, excluding .venv, data,
results and this file. Also catches sosfiltfilt.
"""

from pathlib import Path

from tests.conftest import REPO_ROOT

SKIP_DIRS = {".venv", "data", "results", ".git", ".pytest_cache", "__pycache__"}
NEEDLE = "filt" + "filt"


def test_no_filtfilt_in_repo():
    offenders = []
    for path in REPO_ROOT.rglob("*"):
        if path.suffix not in (".py", ".yaml", ".yml") or path.resolve() == Path(__file__).resolve():
            continue
        if SKIP_DIRS & set(path.relative_to(REPO_ROOT).parts):
            continue
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if NEEDLE in line:
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{n}")
    assert not offenders, offenders
