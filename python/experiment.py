"""Experiment config loading and results directory (results/<date>_<hash>/)."""

import datetime as dt
import hashlib
import json
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"


def load_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def config_hash(cfg: dict) -> str:
    blob = json.dumps(cfg, sort_keys=True, ensure_ascii=True).encode()
    return hashlib.sha1(blob).hexdigest()[:8]


def make_results_dir(cfg: dict, suffix: str = "") -> Path:
    name = f"{dt.date.today().isoformat()}_{config_hash(cfg)}{suffix}"
    out = RESULTS_DIR / name
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    return out


class Tee:
    """Print to stdout and append to a log file."""

    def __init__(self, path: Path):
        self.f = open(path, "w", encoding="utf-8")

    def __call__(self, *args):
        line = " ".join(str(a) for a in args)
        print(line)
        self.f.write(line + "\n")
        self.f.flush()

    def close(self):
        self.f.close()
