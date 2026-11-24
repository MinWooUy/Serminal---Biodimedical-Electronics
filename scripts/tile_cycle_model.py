"""Analytical model of the cycles one beat takes on the PE array, fitted to the board, to compare array shapes and loaders.

    python scripts/tile_cycle_model.py                 # calibration + tables
    python scripts/tile_cycle_model.py --shape 6x8     # one shape, all loader scenarios

Model (rows = output positions, columns = output channels, one tile = R_u x C_u outputs):
    tiles   N   = ceil(T / R_eff) * ceil(Cout / C),  R_eff = R & ~1 for a pooling layer, 3 for fc1 (row_combine)
    cycles  tau = load + compute + DELTA
            load    = ceil(K / W) * max(R_u, C_u)        W = bytes read per cycle (1 today)
            compute = K + R_u + C_u
            serial loader: tau = load + compute + DELTA;  overlapped loader: tau = max(load, compute) + DELTA
    PE utilisation = MAC / (cycles * R * C);  loader-limited closed form: occupancy / (max(R, C) + 1)

DELTA (fixed cycles per tile: post-process, writer, bank swap) is fitted on the 7 x 7 numbers measured on the board on
24/11/2026 (367 091 cycles per beat); the model gives 363 738 (-0.9 %).  Other array shapes and the wide / overlapped
loaders are PROJECTIONS: nothing here has been synthesised or run on the board.  LE per PE (557) comes from the Quartus
report of the 7 x 7 design and is only a rough budget check.
"""
import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from host import ecg_layer_host_pc as eh  # noqa: E402

DELTA = 55                       # fixed cycles per tile, fitted on the board
LE_REST, LE_PER_PE = 14040, 557  # Quartus 21/11: everything but the PE array, and logic elements per PE
LE_TOTAL = 49_760
MEASURED = {"conv0": 21_429, "conv1": 104_379, "conv2": 124_603, "conv3": 113_321, "fc1": 3_137, "fc2": 222}   # board 24/11
SCENARIOS = (("current loader (1 B/cycle, load then compute)", 1, False),
             ("wide read x4", 4, False),
             ("wide read x4 + load overlapped with compute", 4, True),
             ("wide read x8 + load overlapped with compute", 8, True))


def layers():
    out = []
    for layer in eh.load_model(eh.REPO_ROOT):
        sp = layer.spec
        out.append({"name": sp["name"], "T": sp["out_len"], "C": sp["cout"], "K": layer.k_len, "pool": bool(sp["pool"]),
                    "rc": bool(sp["flags"] & 4)})
    return out


def layer_cycles(L, R, C, wide=1, overlap=False):
    if L["pool"]:
        r_eff = max(R & ~1, 2)
    elif L["rc"]:
        if R < 3:
            return None
        r_eff = 3
    else:
        r_eff = R
    total, t_left = 0, L["T"]
    while t_left > 0:
        ru = min(r_eff, t_left)
        t_left -= ru
        c_left = L["C"]
        while c_left > 0:
            cu = min(C, c_left)
            c_left -= cu
            load = math.ceil(L["K"] / wide) * max(ru, cu)
            comp = L["K"] + ru + cu
            total += (max(load, comp) if overlap else load + comp) + DELTA
    return total


def beat_cycles(model, R, C, wide=1, overlap=False):
    per = {L["name"]: layer_cycles(L, R, C, wide, overlap) for L in model}
    return (None, per) if any(v is None for v in per.values()) else (sum(per.values()), per)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shape", help="one array shape RxC, e.g. 6x8 (default: a table of candidates)")
    args = ap.parse_args()
    model = layers()
    macs = sum(L["C"] * L["K"] * (1 if L["rc"] else L["T"]) for L in model)
    total, per = beat_cycles(model, 7, 7)
    meas = sum(MEASURED.values())
    print(f"MAC per beat {macs:,}; model 7x7 {total:,} cycles vs board {meas:,} ({100 * (total / meas - 1):+.1f} %)")
    print("  per layer (model / board): " + ", ".join(f"{k} {v:,}/{MEASURED[k]:,}" for k, v in per.items()))
    shapes = ([tuple(int(x) for x in args.shape.lower().split("x"))] if args.shape else
              [(7, 7), (6, 8), (8, 6), (6, 6), (3, 10), (10, 3), (5, 10), (4, 12), (12, 4), (8, 8), (4, 4)])
    for name, wide, ov in SCENARIOS:
        base = beat_cycles(model, 7, 7, wide, ov)[0]
        print(f"\n== {name}")
        print(f"{'R x C':>7} {'PE':>4} {'LE est':>8} {'fits':>5} {'cycles':>10} {'ms@50':>7} {'vs 7x7':>7} {'PE util':>8}")
        rows = []
        for R, C in shapes:
            t, _ = beat_cycles(model, R, C, wide, ov)
            if t is not None:
                rows.append((t, R, C))
        for t, R, C in sorted(rows):
            le = LE_REST + LE_PER_PE * R * C
            print(f"{R:>3}x{C:<3} {R * C:>4} {le:>8,} {'yes' if le <= 0.95 * LE_TOTAL else 'NO':>5} {t:>10,} "
                  f"{t / 50e3:>7.2f} {t / base:>7.2f} {100 * macs / (t * R * C):>7.1f}%")


if __name__ == "__main__":
    main()
