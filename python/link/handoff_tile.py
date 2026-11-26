"""Host of the tile hand-off package (stand-alone: numpy + pyserial, no torch, no repository). Shipped as host/handoff_tile.py.

    python host/handoff_tile.py status  (--port COM5 | --sim)
    python host/handoff_tile.py smoke   (--port COM5 | --sim)
    python host/handoff_tile.py layer   (--port COM5 | --sim) [--upto conv0] [--beats 0:1]
    python host/handoff_tile.py layers  (--port COM5 | --sim) [--beats 0:80]
    python host/handoff_tile.py golden  (--port COM5 | --sim) [--beats 0:20 | --all]

status / smoke : GET_STATUS flags; the 3x3 example of the FPGA team's README.
layer  : the 80 beats of golden/layers.npz (20 per class N/S/V/F) up to one conv layer, every layer bit for bit.
layers : the same beats through the whole CNN, every layer + logits + label bit for bit.
golden : the 1 306 beats of golden/beats.npz, 4 logits + label bit for bit (~22.5 s per beat on the board at 115 200 baud).
Results: <out>/log.txt, <out>/result.json, <out>/raw_bytes.txt (default out = ket_qua/<date_time>_<command>/ in the package).
Exit code 0 only if everything matched and no beat failed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

try:
    from python.link.line_faults import LineFaults
    from python.link.serial_link import RawLog, SerialTransport, utf8_console
    from python.link.tile_fpga_sim import TileFpgaSim, TileLink
    from python.link.tile_host import LAYERS, TileHost, TileLinkError, run_golden, smoke
except ImportError:                       # stand-alone copy in the hand-off package
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from line_faults import LineFaults
    from serial_link import RawLog, SerialTransport, utf8_console
    from tile_fpga_sim import TileFpgaSim, TileLink
    from tile_host import LAYERS, TileHost, TileLinkError, run_golden, smoke

CONV_KEYS = ("w_q", "b_q", "k", "a_in", "a_out", "pool")
FC1_KEYS = ("wg_q", "wr_q", "b_q", "Eg", "Er", "E", "a_gap", "a_out")
FC2_KEYS = ("w_q", "b_q", "k", "a_in")


def model_arrays(qm) -> dict:
    """Frozen QuantModel -> flat dict for params/model.npz."""
    out = {}
    for i, layer in enumerate(qm.convs):
        for k in CONV_KEYS:
            out[f"conv{i}.{k}"] = np.asarray(layer[k])
    for k in FC1_KEYS:
        out[f"fc1.{k}"] = np.asarray(qm.fc1[k])
    for k in FC2_KEYS:
        out[f"fc2.{k}"] = np.asarray(qm.fc2[k])
    return out


def load_model(path: Path):
    """params/model.npz -> object with .convs / .fc1 / .fc2 as tile_sched.run_beat expects."""
    z = np.load(path)
    n = len({k.split(".")[0] for k in z.files if k.startswith("conv")})
    convs = []
    for i in range(n):
        d = {k: z[f"conv{i}.{k}"] for k in CONV_KEYS}
        for k in ("k", "a_in", "a_out"):
            if d[k].ndim == 0:
                d[k] = int(d[k])
        d["pool"] = bool(d["pool"])
        convs.append(d)
    fc1 = {k: (int(z[f"fc1.{k}"]) if z[f"fc1.{k}"].ndim == 0 else z[f"fc1.{k}"]) for k in FC1_KEYS}
    fc2 = {k: (int(z[f"fc2.{k}"]) if z[f"fc2.{k}"].ndim == 0 else z[f"fc2.{k}"]) for k in FC2_KEYS}
    return SimpleNamespace(convs=convs, fc1=fc1, fc2=fc2)


def parse_beats(spec: str, n: int) -> list[int]:
    a, _, b = spec.partition(":")
    lo, hi = int(a or 0), int(b) if b else int(a) + 1
    if not 0 <= lo < hi <= n:
        raise SystemExit(f"--beats {spec}: chỉ có nhịp 0..{n - 1}")
    return list(range(lo, hi))


class _Log:
    def __init__(self, path: Path):
        self.f = open(path, "w", encoding="utf-8")

    def __call__(self, *a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        self.f.write(s + "\n")
        self.f.flush()


def main(argv=None, transport=None) -> int:
    utf8_console()
    pkg = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=("status", "smoke", "layer", "layers", "golden"))
    ap.add_argument("--port")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--sim", action="store_true", help="simulated accelerator in this process (no board)")
    ap.add_argument("--timeout", type=float, default=1.0)
    ap.add_argument("--beats")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--upto", default="conv0", choices=LAYERS[:7])
    ap.add_argument("--pkg", default=str(pkg))
    ap.add_argument("--out")
    ap.add_argument("--flip", type=float, default=0.0)
    ap.add_argument("--drop", type=float, default=0.0)
    ap.add_argument("--lose", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    pkg = Path(args.pkg)
    out = Path(args.out) if args.out else pkg / "ket_qua" / f"{time.strftime('%Y%m%d_%H%M%S')}_{args.command}"
    out.mkdir(parents=True, exist_ok=True)
    log = _Log(out / "log.txt")
    if transport is None:
        if args.sim:
            f = LineFaults(args.flip, args.drop, args.lose)
            transport, desc = TileLink(TileFpgaSim(), down=f, up=f, seed=args.seed), "bộ tăng tốc GIẢ LẬP (đồng hồ ảo)"
        elif args.port:
            transport, desc = SerialTransport(args.port, args.baud), f"{args.port} @ {args.baud} 8N1"
        else:
            raise SystemExit("cần --port COMx (board) hoặc --sim (giả lập)")
    else:
        desc = type(transport).__name__
    raw = RawLog(transport, out / "raw_bytes.txt", header=desc, proto="tile protocol (docs/proto_tile.md)")
    host = TileHost(raw, timeout=args.timeout)
    log(f"đường truyền: {desc}; kết quả -> {out}")
    result: dict = {"command": args.command}
    ok = False
    try:
        if args.command == "status":
            result["status"] = host.status()
            log(f"GET_STATUS: {result['status']}")
            ok = True
        elif args.command == "smoke":
            result["smoke"] = s = smoke(host)
            log(f"ô 3x3: nhận {s['raw']}, mong đợi {s['expected']} -> {'ĐẠT' if s['ok'] else 'KHÔNG ĐẠT'}")
            ok = s["ok"]
        else:
            qm = load_model(pkg / "params" / "model.npz")
            src = "beats.npz" if args.command == "golden" else "layers.npz"
            g = dict(np.load(pkg / "golden" / src))
            n = len(g["z_q"])
            default = {"golden": "0:20", "layers": f"0:{n}", "layer": "0:1"}[args.command]
            beats = list(range(n)) if args.all else parse_beats(args.beats or default, n)
            upto = args.upto if args.command == "layer" else None
            log(f"{args.command}: {len(beats)} nhịp của golden/{src} ({beats[0]}..{beats[-1]})" + (f", tới {upto}" if upto else ""))
            r = run_golden(host, qm, g, beats, log, upto=upto)
            result["golden"] = r
            log(f"{r['exact']}/{r['beats']} nhịp trùng từng bit" + ("" if upto else f", nhãn đúng {r['pred_ok']}/{r['beats']}")
                + f"; {len(r['failed'])} nhịp lỗi đường truyền; {r['transactions']} giao dịch, {r['bytes']} B, "
                f"{r['seconds']:.1f} s ({r['seconds'] / max(r['beats'], 1):.2f} s / nhịp)")
            ok = r["exact"] == r["beats"] and not r["failed"]
            log("ĐẠT" if ok else "KHÔNG ĐẠT")
    except TileLinkError as e:
        log(f"LỖI ({e.kind}): {e}")
        result["error"] = {"kind": e.kind, "message": str(e)}
    result["link"] = vars(host.stats)
    (out / "result.json").write_text(json.dumps(result, indent=1, ensure_ascii=False,
                                                default=lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o)),
                                     encoding="utf-8")
    raw.close()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
