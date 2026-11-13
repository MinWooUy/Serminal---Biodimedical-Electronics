#!/usr/bin/env python3
"""Shim giữ đường dẫn cũ: mã thật nằm ở host/ecg_host.py (xem docs/restructure-plan.md).

Lệnh cũ như `py -3 rtl/cnn1d/comm/host/ecg_host.py COM4 --full` vẫn chạy.
"""

import pathlib
import runpy

HOST_SCRIPT = pathlib.Path(__file__).resolve().parents[4] / "host" / "ecg_host.py"

if __name__ == "__main__":
    runpy.run_path(str(HOST_SCRIPT), run_name="__main__")
