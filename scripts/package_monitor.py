"""Gom app xem tin hieu (scripts/board_monitor.py + scripts/live_monitor.py) thanh 1 folder doc lap de gui nguoi khac.

    python scripts/package_monitor.py [--out dist/ecg_monitor] [--zip]

Folder tao ra chi chua: live_monitor.py + cac module python/ no import + du lieu golden + mo hinh int8 da dong bang
+ config. KHONG can torch / wfdb (hai thu nay chi dung luc train / doc MIT-BIH goc); ban sao duoc va 2 cho import
torch/wfdb thanh tuy chon. Ban goc trong repo KHONG bi sua.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

PY_FILES = [
    "python/__init__.py", "python/experiment.py",
    "host/__init__.py", "host/board_classifier.py", "host/ecg_layer_host_pc.py",
    "scripts/live_monitor.py", "scripts/board_monitor.py",
    "configs/beats.yaml", "configs/detector_pt.yaml",
    "export/final_models_manifest.json", "export/golden_manifest.json", "export/weights.h",
]
PY_DIRS = ["python/data", "python/dsp", "python/quant", "python/viz", "python/link"]     # copy nguyen thu muc

# (file, text goc, text thay) - assert text goc ton tai de bao loi ngay neu nguon thay doi
PATCHES = [
    ("python/quant/int8.py",
     "from dataclasses import dataclass, field\n\nimport numpy as np\nimport torch\n\nfrom python.models.cnn1d import ECGCNN\n",
     "from __future__ import annotations\n\nfrom dataclasses import dataclass, field\n\nimport numpy as np\n\n"
     "try:                                    # torch chi can khi train/quantize; suy luan int8 chi dung numpy\n"
     "    import torch\n    from python.models.cnn1d import ECGCNN\n"
     "except ImportError:\n    torch = ECGCNN = None\n"),
    ("python/data/mitbih.py",
     "import numpy as np\nimport wfdb\n",
     "import numpy as np\n\ntry:                                    # wfdb chi can khi doc MIT-BIH goc\n"
     "    import wfdb\nexcept ImportError:\n    wfdb = None\n"),
    # board_monitor lay quantise_rr_parts qua scripts.pipeline_int_eval (keo theo torch + train code);
    # ham nay nam o python.quant.pipeline_int tu commit 8edac9b
    ("scripts/board_monitor.py",
     "from scripts.pipeline_int_eval import quantise_rr_parts",
     "from python.quant.pipeline_int import quantise_rr_parts"),
]

REQUIREMENTS = "numpy\nscipy\nmatplotlib\npyyaml\npyserial\n"

RUN_BAT = r"""@echo off
rem Chay app xem tin hieu ECG. Lan dau tu tao .venv va cai thu vien (can Python 3.10+ va internet).
cd /d "%~dp0"
if not exist .venv (
    echo [setup] tao moi truong ao .venv ...
    python -m venv .venv || (echo Khong tim thay Python. Cai Python 3.10+ va tick "Add to PATH". & pause & exit /b 1)
    .venv\Scripts\python -m pip install -q -r requirements.txt || (echo Cai thu vien that bai & pause & exit /b 1)
)
if "%~1"=="" (
    echo Cach dung:
    echo   run_monitor.bat demo             ^(khong can board: FPGA gia lap bang phan mem^)
    echo   run_monitor.bat check            ^(kiem tra bit-exact bang FPGA gia lap, khong cua so^)
    echo   run_monitor.bat COM5             ^(board that, doi COM5 theo may^)
    echo   run_monitor.bat COM5 check       ^(board that, kiem tra bit-exact rec208^)
    pause & exit /b 0
)
if /i "%~1"=="demo"  (.venv\Scripts\python scripts\board_monitor.py --sim --sim-pace 1 --golden rec208_237600 & goto :eof)
if /i "%~1"=="check" (.venv\Scripts\python scripts\board_monitor.py --sim --check rec208_237600 & pause & goto :eof)
if /i "%~2"=="check" (.venv\Scripts\python scripts\board_monitor.py --serial %1 --check rec208_237600 & pause & goto :eof)
.venv\Scripts\python scripts\board_monitor.py --serial %1 --golden rec208_237600
"""

README = """# ECG board monitor (ban doc lap)

App xem tin hieu ECG da loc 0.5-40 Hz, danh dau R-peak theo nhan N/S/V/F va xac suat lop.
- `scripts/board_monitor.py`: PC tien xu ly (loc, Pan-Tompkins, cua so nhip, z-score), moi nhip duoc **FPGA** phan loai
  qua UART (giao thuc lenh-tung-lop `layer_cmd_controller`); cua so hien nhan do board tra ve. Co the thay board bang
  FPGA gia lap bang phan mem (`--sim`).
- `scripts/live_monitor.py`: ban cu, chi giai ma luong khung UART SAMPLES/BEAT/STATS (`docs/uart_protocol.md`).

## Cai dat (1 lan)
Can Python 3.10+ (tick "Add Python to PATH" khi cai). Chi can double-click `run_monitor.bat`,
no tu tao `.venv` va cai `requirements.txt` (numpy, scipy, matplotlib, pyyaml, pyserial). Khong can torch / wfdb.

Thu cong:
```
python -m venv .venv
.venv\\Scripts\\pip install -r requirements.txt
```

## Chay
| Muc dich | Lenh |
|---|---|
| Demo khong can board (FPGA gia lap) | `run_monitor.bat demo` hoac `python scripts/board_monitor.py --sim --sim-pace 1 --golden rec208_237600` |
| Kiem tra bit-exact bang gia lap, khong cua so | `run_monitor.bat check` |
| Board that, cua so live | `run_monitor.bat COM5` hoac `python scripts/board_monitor.py --serial COM5 --golden rec208_237600` |
| Board that, kiem tra bit-exact | `run_monitor.bat COM5 check` |
| Doi toc do / baud | them `--speed 1` (1 = thoi gian thuc) / `--baud 921600` (mac dinh 1500000, phai khop bitstream) |
| App cu doc khung UART (mo phong thiet bi) | `python scripts/live_monitor.py --golden rec208_237600 [--corrupt 0.02]` |

Doan golden co san: rec101_93600, rec118_237600, rec119_489600, rec201_522000, rec207_604800, rec208_237600,
rec209_198000, rec223_187200.

Cong COM xem trong Device Manager > Ports. Dong cua so khac dang giu cong (Quartus, terminal...) truoc khi chay.

## Luu y
- Board that: nap bitstream (.sof) co `layer_cmd_controller` + FC1/FC2/argmax truoc, baud phai khop (1500000 hoac 921600).
- Moi nhip mat ~0.3-1 s tren day UART, nen nhan hien tre hon duong ECG (chay thread rieng).
- Khong dong goi `board_eval.py` / `board_compare.py` (can torch + du lieu MIT-BIH goc `data/raw`, ~83 MB).
- Folder nay duoc sinh boi `scripts/package_monitor.py` tu repo goc; khi code goc doi, chay lai script de cap nhat.
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dist/ecg_monitor")
    ap.add_argument("--zip", action="store_true", help="nen them thanh .zip canh folder")
    args = ap.parse_args()
    out = (REPO / args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    for child in out.iterdir():                     # xoa noi dung, giu thu muc (co the dang mo trong shell)
        shutil.rmtree(child) if child.is_dir() else child.unlink()

    man = json.loads((REPO / "export/final_models_manifest.json").read_text(encoding="utf-8"))
    lite = f"results/{man['lite']['dir']}"
    files = PY_FILES + [f"{lite}/config.yaml", f"{lite}/lite_int8.npz"]
    for rel in files:
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / rel, dst)
    shutil.copytree(REPO / "export/golden", out / "export/golden")
    for rel in PY_DIRS:
        shutil.copytree(REPO / rel, out / rel, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

    for rel, old, new in PATCHES:
        p = out / rel
        s = p.read_text(encoding="utf-8")
        if old not in s:
            sys.exit(f"patch that bai: khong thay doan can thay trong {rel} (nguon da doi?)")
        p.write_text(s.replace(old, new, 1), encoding="utf-8", newline="\n")

    (out / "requirements.txt").write_text(REQUIREMENTS, encoding="utf-8")
    (out / "run_monitor.bat").write_text(RUN_BAT, encoding="ascii", newline="\r\n")
    (out / "README.md").write_text(README, encoding="utf-8")
    print(f"da tao {out}")
    if args.zip:
        z = shutil.make_archive(str(out), "zip", root_dir=out.parent, base_dir=out.name)
        print(f"da nen {z}")


if __name__ == "__main__":
    main()
