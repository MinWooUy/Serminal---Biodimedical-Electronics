"""Build the record-grouped 5-fold split on DS1 and save it to configs/splits/ds1_k5.json."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from python.data.mitbih import (  # noqa: E402
    DEFAULT_RAW_DIR, check_ds1_sanity, count_labels, load_record,
)
from python.data.splits import (  # noqa: E402
    DEFAULT_FOLDS_PATH, fold_counts, get_ds1_records, make_group_kfold, save_folds,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(DEFAULT_RAW_DIR))
    ap.add_argument("--out", default=str(DEFAULT_FOLDS_PATH))
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-trials", type=int, default=20000)
    args = ap.parse_args()

    records = [load_record(r, args.dir) for r in get_ds1_records()]
    check_ds1_sanity(count_labels(records))  # refuse to build folds on bad data
    per_record = {rec.rec_id: count_labels([rec]) for rec in records}

    folds = make_group_kfold(per_record, k=args.k, seed=args.seed, n_trials=args.n_trials)
    save_folds(folds, per_record, args.out, seed=args.seed, n_trials=args.n_trials)

    for i, (f, c) in enumerate(zip(folds, fold_counts(folds, per_record))):
        print(f"fold {i}: records={f}")
        print(f"        N={c['N']} S={c['S']} V={c['V']} F={c['F']} Q={c['Q']}")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
