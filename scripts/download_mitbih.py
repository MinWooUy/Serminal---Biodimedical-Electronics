"""Download MIT-BIH Arrhythmia (non-paced records) into data/raw/mitdb."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from python.data.mitbih import DEFAULT_RAW_DIR, download  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(DEFAULT_RAW_DIR))
    args = ap.parse_args()
    download(args.dir)
    n = len(list(Path(args.dir).glob("*.hea")))
    print(f"{n} header files in {args.dir}")


if __name__ == "__main__":
    main()
