"""Print DS1 beat counts per record/class, VF segments, and the sanity gate result."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from python.data.constants import DS1_EXPECTED_COUNTS, SANITY_TOLERANCE  # noqa: E402
from python.data.mitbih import (  # noqa: E402
    DEFAULT_RAW_DIR, check_ds1_sanity, count_labels, load_record,
)
from python.data.splits import get_ds1_records  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(DEFAULT_RAW_DIR))
    args = ap.parse_args()

    records = [load_record(r, args.dir) for r in get_ds1_records()]
    print(f"{'rec':>4} {'N':>6} {'S':>5} {'V':>5} {'F':>5} {'Q':>4} {'VFseg':>5}")
    for rec in records:
        c = count_labels([rec])
        print(f"{rec.rec_id:>4} {c['N']:>6} {c['S']:>5} {c['V']:>5} {c['F']:>5} "
              f"{c['Q']:>4} {len(rec.vf_segments):>5}")
    total = count_labels(records)
    print(f"{'sum':>4} {total['N']:>6} {total['S']:>5} {total['V']:>5} {total['F']:>5} "
          f"{total['Q']:>4}")
    print("expected:", DS1_EXPECTED_COUNTS)
    try:
        dev = check_ds1_sanity(total)
        print(f"SANITY OK (tol {SANITY_TOLERANCE:.0%}):",
              {c: f"{d:+.2%}" for c, d in dev.items()})
    except ValueError as e:
        print("SANITY FAILED:", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
