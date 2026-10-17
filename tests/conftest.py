from pathlib import Path

import pytest

from python.data.constants import DS1
from python.data.mitbih import DEFAULT_RAW_DIR

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = DEFAULT_RAW_DIR


def _ds1_available() -> bool:
    return all((RAW_DIR / f"{r}.{ext}").exists() for r in DS1 for ext in ("hea", "dat", "atr"))


def pytest_collection_modifyitems(config, items):
    if _ds1_available():
        return
    skip = pytest.mark.skip(reason=f"MIT-BIH DS1 not found in {RAW_DIR}")
    for item in items:
        if "needs_data" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def raw_dir() -> Path:
    return RAW_DIR


@pytest.fixture(scope="session")
def ds1_records(raw_dir):
    from python.data.mitbih import load_record
    return {r: load_record(r, raw_dir) for r in DS1}
