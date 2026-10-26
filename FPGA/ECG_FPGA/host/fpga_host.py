"""Stand-alone PC host for the FPGA team (shipped as host/fpga_host.py in the hand-off package).

Needs only Python 3.10+ and pyserial; uses the package files, not the project:
    params/ecg_params.vh        -> expected WEIGHTS_CRC32
    golden/uart_in.bin          -> the 1 306 BEAT_IN frames (beat_id 0..1305)
    golden/result_expected.hex  -> per beat: beat_id, logit0..3, pred (32-bit words)

    python host/fpga_host.py loopback --port COM5     TX shorted to RX on the USB-UART module (no FPGA)
    python host/fpga_host.py ping     --port COM5     PING -> PONG, checks proto_ver, mode, weights_crc32
    python host/fpga_host.py golden   --port COM5     1 306 beats, logits + pred compared bit for bit
    options: --baud 115200, --raw-log FILE (every byte with a time stamp), --limit N, --out DIR
Results: <out>/log.txt and <out>/result.json (default out = ket_qua/<date_time>_<command>/ in the package).
Protocol: PROTO_V2.md (copy of docs/proto_v2.md).
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

try:
    from python.link import proto_v2 as pv2
    from python.link.host import Host, LinkError
    from python.link.serial_link import (
        RawLog, SerialTransport, check_pong, describe_pong, diagnose_ping, loopback, utf8_console,
    )
except ImportError:                       # inside the hand-off package: host/ holds the copies
    import proto_v2 as pv2
    from link_host import Host, LinkError
    from serial_link import RawLog, SerialTransport, check_pong, describe_pong, diagnose_ping, loopback, utf8_console


def expected_crc(pkg: Path) -> int:
    text = (pkg / "params" / "ecg_params.vh").read_text(encoding="ascii")
    return int(re.search(r"WEIGHTS_CRC32 = 32'h([0-9A-Fa-f]{8})", text).group(1), 16)


def load_golden(pkg: Path) -> tuple[list, list]:
    beats = pv2.Decoder(accept=pv2.HOST_TO_FPGA).feed((pkg / "golden" / "uart_in.bin").read_bytes())
    if not beats or not all(isinstance(b, pv2.BeatIn) for b in beats):
        raise SystemExit("golden/uart_in.bin: not a sequence of BEAT_IN frames")
    words = [int(t, 16) for t in (pkg / "golden" / "result_expected.hex").read_text(encoding="ascii").split()]
    signed = [w - (1 << 32) if w >= 1 << 31 else w for w in words]
    exp = [tuple(signed[i:i + 6]) for i in range(0, len(signed), 6)]
    if len(exp) != len(beats) or any(e[0] != b.beat_id for e, b in zip(exp, beats)):
        raise SystemExit("golden/result_expected.hex does not match golden/uart_in.bin")
    return beats, exp


def run_golden(host: Host, beats: list, exp: list, log=print) -> dict:
    ok_logits = ok_pred = failed = 0
    sends = {str(k): 0 for k in range(1, pv2.MAX_RETRIES + 2)}
    cycles, mismatches = [], []
    for b, e in zip(beats, exp):
        try:
            res, k = host.classify(b.beat_id, b.z_q, b.rr_q)
        except LinkError:
            failed += 1
            mismatches.append({"beat_id": b.beat_id, "error": "no RESULT after 4 sends"})
            continue
        sends[str(k)] += 1
        cycles.append(res.cycles)
        lo, po = tuple(res.logits) == tuple(e[1:5]), res.pred == e[5]
        ok_logits += lo
        ok_pred += po
        if not (lo and po):
            mismatches.append({"beat_id": b.beat_id, "fpga": [*res.logits, res.pred], "expected": list(e[1:])})
        if (b.beat_id + 1) % 100 == 0:
            log(f"  ... {b.beat_id + 1}/{len(beats)} beats, {len(mismatches)} problems so far")
    n = len(beats)
    return {"beats": n, "logits_exact": ok_logits, "pred_exact": ok_pred, "failed": failed,
            "pass": failed == 0 and ok_logits == n and ok_pred == n, "sends_histogram": sends,
            "cycles": ({"min": min(cycles), "median": sorted(cycles)[len(cycles) // 2], "max": max(cycles)}
                       if cycles else None),
            "link": {**host.stats.as_dict(), "rx_bytes": host.rx_bytes,
                     "decoder": {"frames_ok": host.dec.frames_ok, "bad_crc": host.dec.bad_crc,
                                 "bad_len": host.dec.bad_len, "bad_type": host.dec.bad_type,
                                 "skipped_bytes": host.dec.skipped_bytes}},
            "mismatches": mismatches[:50]}


class _Log:
    def __init__(self, path: Path):
        self.f = open(path, "w", encoding="utf-8")

    def __call__(self, *a):
        line = " ".join(str(x) for x in a)
        print(line)
        self.f.write(line + "\n")
        self.f.flush()


def main(argv=None, transport=None) -> int:
    """Returns the exit code. `transport` (tests) replaces the serial port."""
    utf8_console()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("loopback", "ping", "golden"))
    ap.add_argument("--port", help="COM port of the USB-UART module, e.g. COM5 (or a pyserial URL such as loop://)")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--raw-log", help="write every byte sent / received with a time stamp to this file")
    ap.add_argument("--limit", type=int, help="golden: only the first N beats")
    ap.add_argument("--out", help="directory for log.txt / result.json")
    ap.add_argument("--pkg", default=str(Path(__file__).resolve().parents[1]), help="hand-off package root")
    args = ap.parse_args(argv)
    pkg = Path(args.pkg)
    out = Path(args.out) if args.out else pkg / "ket_qua" / f"{time.strftime('%Y%m%d_%H%M%S')}_{args.command}"
    out.mkdir(parents=True, exist_ok=True)
    log = _Log(out / "log.txt")
    if transport is None:
        if not args.port:
            log("Thiếu --port (ví dụ --port COM5).")
            return 2
        transport = SerialTransport(args.port, args.baud)
        desc = f"serial {args.port} @ {args.baud} 8N1"
    else:
        desc = "injected transport (test)"
    raw = args.raw_log or str(out / "raw_bytes.txt")
    transport = RawLog(transport, raw, header=f"{args.command}; {desc}")
    log(f"link: {desc}; raw byte log -> {raw}; results -> {out}")
    host = Host(transport)
    result = {"command": args.command, "link": desc}
    try:
        if args.command == "loopback":
            r = loopback(transport)
            for line in r["messages"]:
                log(line)
            result.update(r)
            return 0 if r["ok"] else 1
        crc = expected_crc(pkg)
        try:
            pong = host.ping()
        except LinkError as e:
            log(f"KHÔNG CÓ PONG: {e}")
            for line in diagnose_ping(host, crc):
                log(line)
            result.update({"pong": None, "rx_bytes": host.rx_bytes, "rx_head": bytes(host.rx_head[:64]).hex(" ")})
            return 2
        log(describe_pong(pong))
        result["pong"] = {"proto_ver": pong.proto_ver, "mode": pong.mode, "build_id": f"0x{pong.build_id:08X}",
                          "weights_crc32": f"0x{pong.weights_crc32:08X}"}
        bad = check_pong(pong, crc)
        if bad:
            log("PONG KHÔNG KHỚP: " + "; ".join(bad))
            for line in diagnose_ping(host, crc, pong):
                log(line)
            return 3
        log("PING ĐẠT.")
        if args.command == "ping":
            return 0
        beats, exp = load_golden(pkg)
        if args.limit:
            beats, exp = beats[:args.limit], exp[:args.limit]
        log(f"golden: gửi {len(beats)} nhịp ...")
        r = run_golden(host, beats, exp, log)
        result.update(r)
        log(f"golden: {r['logits_exact']}/{r['beats']} logit trùng từng bit, {r['pred_exact']}/{r['beats']} nhãn, "
            f"{r['failed']} nhịp không có RESULT -> {'ĐẠT' if r['pass'] else 'TRƯỢT'}")
        log(f"số lần gửi mỗi nhịp: {r['sends_histogram']}; cycles: {r['cycles']}")
        log(f"đường truyền: {r['link']}")
        for m in r["mismatches"][:10]:
            log(f"  SAI: {m}")
        return 0 if r["pass"] else 1
    finally:
        (out / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        transport.close()


if __name__ == "__main__":
    sys.exit(main())
