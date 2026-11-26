#!/usr/bin/env python3
"""Minimal board bring-up for the layer_cmd_controller protocol.

Examples:
  python scripts/layer_board_test.py info --port COM5
  python scripts/layer_board_test.py smoke --port COM5

The script writes reports only under results/board/ in this workspace.
It does not implement or assume CONFIG_ALL / LOAD_COMMIT / RUN_ALL / GET_RESULT.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from python.link.layer_host import LayerHost, LayerLinkError  # noqa: E402
from python.link.proto_layer import CMD_NAMES  # noqa: E402
from python.link.serial_link import RawLog, SerialTransport, utf8_console  # noqa: E402


def json_default(value):
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"cannot JSON-encode {type(value).__name__}")


def main(argv=None) -> int:
    utf8_console()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("info", "smoke"))
    parser.add_argument("--port", required=True, help="COM port or /dev/ttyUSBx")
    parser.add_argument("--baud", type=int, default=921600)
    parser.add_argument("--timeout", type=float, default=3.0)
    parser.add_argument("--out", help="result directory (default: results/board/<timestamp>_<command>)")
    args = parser.parse_args(argv)

    out = Path(args.out) if args.out else ROOT / "results" / "board" / f"{time.strftime('%Y%m%d_%H%M%S')}_{args.command}"
    out.mkdir(parents=True, exist_ok=True)
    raw = None
    host = None
    result = {"command": args.command, "port": args.port, "baud": args.baud, "status": "not_started"}
    log_lines = []

    def log(message: str):
        print(message, flush=True)
        log_lines.append(message)

    try:
        serial = SerialTransport(args.port, args.baud)
        raw = RawLog(serial, out / "raw_bytes.txt", header=f"{args.port} @ {args.baud} 8N1",
                     proto="layer_cmd_controller protocol (CMD 0x08..0x15)")
        host = LayerHost(raw, timeout=args.timeout)
        if args.command == "info":
            info = host.info()
            result.update(status="pass", hardware=asdict(info))
            log("INFO from FPGA:")
            for key, value in asdict(info).items():
                log(f"  {key}: {value}")
            log(f"  PE array: {info.rows}x{info.cols}; clock {info.clk_hz} Hz; UART {info.baud} baud")
            log(f"  feature flags: 0x{info.features:02X}; MAX_PAYLOAD={info.max_payload}")
            if info.baud != args.baud:
                log(f"WARNING: CLI baud {args.baud} differs from INFO baud {info.baud}; reconnect using the reported baud.")
        else:
            smoke = host.smoke()
            result.update(status="pass" if smoke["passed"] else "fail", smoke=smoke)
            log(f"Layer smoke: got {smoke['got']}, expected {smoke['expected']}; cycles={smoke['run'].cycles}")
            log("PASS" if smoke["passed"] else "FAIL")
    except (LayerLinkError, ValueError, OSError, ImportError) as exc:
        result.update(status="error", error={"type": type(exc).__name__, "kind": getattr(exc, "kind", None), "message": str(exc)})
        log(f"ERROR: {exc}")
    finally:
        if host is not None:
            result["link_stats"] = asdict(host.stats)
            host.close()
        elif raw is not None:
            raw.close()
        (out / "log.txt").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
        (out / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, default=json_default) + "\n", encoding="utf-8")
        print(f"Results: {out}")

    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
