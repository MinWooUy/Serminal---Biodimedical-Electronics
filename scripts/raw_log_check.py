"""Read a UART raw-byte log of host/ecg_layer_host_pc.py offline: array shape (INFO), cycles of every RUN, and
whether the data read back for every layer matches the golden tensors of that beat.

    python scripts/raw_log_check.py results/hardware_logs/layer0-5_rec101_93600_beat000_20261128-151358.raw.log

The record and beat are taken from the log header ("# raw byte log, layer protocol; <date>; <record> beat <n>").
Packets are re-assembled from the TX and RX byte streams (one RX line can hold part of a packet), and the n-th
response is paired with the n-th command, as the protocol answers every command in order.
"""
import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from host import ecg_layer_host_pc as eh  # noqa: E402

CMD_INFO, CMD_RUN, CMD_READ_GAP, CMD_READ_FMAP = 0x15, 0x0E, 0x0F, 0x10
NAMES = ["conv0", "conv1", "conv2", "conv3", "fc1", "fc2"]


def packets(stream: bytes) -> list[tuple[int, bytes]]:
    out, i = [], 0
    while i + 6 <= len(stream):
        if stream[i:i + 2] != b"\xa5\x5a":
            i += 1
            continue
        cmd, ln = stream[i + 2], stream[i + 4] | stream[i + 5] << 8
        out.append((cmd, stream[i + 6:i + 6 + ln]))
        i += 6 + ln + 2                                  # header 6 B + payload + CRC-16
    return out


def parse(path: Path):
    tx, rx, header = bytearray(), bytearray(), ""
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("#"):
            header = header or line
            continue
        p = line.split()
        if len(p) >= 4:
            (tx if p[1] == "TX" else rx).extend(int(x, 16) for x in p[3:])
    cmds, rsps = packets(bytes(tx)), packets(bytes(rx))
    if len(cmds) != len(rsps):
        raise SystemExit(f"{path.name}: {len(cmds)} commands but {len(rsps)} responses")
    info, layers = None, []
    for (c, _), (_, r) in zip(cmds, rsps):
        if c == CMD_INFO:
            info = r
        elif c == CMD_RUN:
            layers.append({"status": r[0], "cycles": int.from_bytes(r[1:5], "little"), "data": bytearray()})
        elif c in (CMD_READ_GAP, CMD_READ_FMAP) and layers:
            layers[-1]["data"].extend(r)
    m = re.search(r";\s*(\S+) beat (\d+)", header)
    return info, layers, (m.group(1), int(m.group(2))) if m else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("logs", nargs="+", type=Path)
    args = ap.parse_args()
    all_ok = True
    for path in args.logs:
        info, layers, beat = parse(path)
        shape = f"{info[1]}x{info[2]}" if info and len(info) >= 3 else "?"
        gold = eh.load_beat(eh.REPO_ROOT, *beat) if beat else None
        print(f"{path.name}: array {shape}, {beat[0]} beat {beat[1]}" if beat else f"{path.name}: array {shape}")
        for i, L in enumerate(layers):
            if gold is None:
                verdict = "no golden"
            else:
                if i == 3:
                    exp = b"".join((int(v) & 0xFFFF).to_bytes(2, "little") for v in gold.gap)
                else:
                    exp = eh._fmap_expected_bytes(i, gold)
                ok = L["status"] == 0 and bytes(L["data"][:len(exp)]) == exp and len(L["data"]) >= len(exp)
                all_ok &= ok
                verdict = f"{'PASS' if ok else 'FAIL'} {len(exp):,} B"
            print(f"  {NAMES[i]:<6} status {L['status']}  {L['cycles']:>9,} cycles  {verdict}")
        print(f"  total {sum(L['cycles'] for L in layers):,} cycles")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
