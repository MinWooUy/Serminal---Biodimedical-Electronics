"""Write export/final_models_manifest.json: SHA-256 of EVERY file in the two final model directories
(Lite checkpoint + int8 parameters, Full checkpoint, their configs/logs) and of export/weights.h.

scripts/final_eval_ds2.py recomputes these hashes before evaluating DS2 and refuses to run on any mismatch.
The result directories themselves are not versioned (results/ is git-ignored); this manifest is.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from python.data.mitbih import REPO_ROOT  # noqa: E402

MANIFEST = REPO_ROOT / "export" / "final_models_manifest.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dir_hashes(d: Path) -> dict[str, str]:
    return {p.relative_to(d).as_posix(): sha256(p) for p in sorted(d.rglob("*")) if p.is_file()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--final-lite", required=True)
    ap.add_argument("--final-full", required=True)
    args = ap.parse_args()
    fl, ff = Path(args.final_lite), Path(args.final_full)
    for d, must in ((fl, ("model.pt", "lite_int8.npz", "config.yaml")), (ff, ("model.pt", "config.yaml"))):
        missing = [m for m in must if not (d / m).exists()]
        if missing:
            raise SystemExit(f"{d}: missing {missing}")
    manifest = {
        "lite": {"dir": fl.name, "files": dir_hashes(fl)},
        "full": {"dir": ff.name, "files": dir_hashes(ff)},
        "weights_h_sha256": sha256(REPO_ROOT / "export" / "weights.h"),
        "note": "lite_int8.npz is the single source of the integer Lite constants: export/weights.h is generated "
                "from it, and the DS2 'Lite integer' branch loads it (no re-quantisation).",
    }
    MANIFEST.write_bytes((json.dumps(manifest, indent=2) + "\n").encode("utf-8"))
    print(f"{MANIFEST}: lite {len(manifest['lite']['files'])} files, full {len(manifest['full']['files'])} files; "
          f"weights.h {manifest['weights_h_sha256'][:16]}...")


if __name__ == "__main__":
    main()
