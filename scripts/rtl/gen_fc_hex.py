"""Sinh vector cho tb_fc_layers.v (FC1 row_combine + gap_in, FC2 out_int32 + argmax).

Chay tu goc repo:  python scripts/rtl/gen_fc_hex.py
Ghi vao rtl/cnn1d/sim/hex/fc_*.hex ($readmemh, moi dong mot gia tri).

Truoc khi ghi, script tinh lai FC1/FC2 theo dung cach phan cung lam (3 hang PE: lo, hi, aux,
moi hang da preload +bias) va khang dinh trung golden tung bit, de testbench chi con kiem RTL.
"""
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
GOLD = ROOT / "export" / "golden"
OUT = ROOT / "rtl" / "cnn1d" / "sim" / "hex"

# mot nhip dau tien moi lop (docs/cnn_explained.md muc 6): N, S, V (doan sai -> S), F (doan sai -> N)
BEATS = [("rec101_93600", 0), ("rec101_93600", 121), ("rec118_237600", 100), ("rec201_522000", 11)]


def c_array(header, name):
    m = re.search(r"\b" + name + r"\b[^=]*=\s*\{(.*?)\};", header, re.S)
    if not m:
        raise RuntimeError(f"{name} not found in weights.h")
    return np.array([int(v) for v in re.findall(r"-?\d+", m.group(1))], dtype=np.int64)


def rsr(x, s):
    s = np.asarray(s, dtype=np.int64)
    return np.where(s > 0, (x + (np.int64(1) << np.maximum(s - 1, 0))) >> s, x)


def load(seg, name):
    meta = next(x for x in json.loads((GOLD / "manifest.json").read_text())["segments"] if x["name"] == seg)
    a = meta["arrays"][name]
    return np.fromfile(GOLD / a["file"], dtype=a["dtype"]).reshape(a["shape"]).astype(np.int64)


def write(name, values, digits):
    mask = (1 << (4 * digits)) - 1
    (OUT / name).write_text("".join(f"{int(v) & mask:0{digits}x}\n" for v in values))


def main():
    h = (ROOT / "export" / "weights.h").read_text()
    WG = c_array(h, "FC1_WG").reshape(32, 64)
    WR = c_array(h, "FC1_WR").reshape(32, 4)
    B1 = c_array(h, "FC1_B")
    SG = c_array(h, "FC1_SHIFT_G")
    SR = c_array(h, "FC1_SHIFT_R")
    SO = c_array(h, "FC1_SHIFT_OUT")
    W2 = c_array(h, "FC2_W").reshape(4, 32)
    B2 = c_array(h, "FC2_B")
    assert (SR == 0).all(), "row_combine gia dinh SHIFT_R = 0 (khoi RR cong thang)"
    W1 = np.concatenate([WG, WR], axis=1)                       # cot trong so [o][68]

    gaps, rrs, hs, logits, preds = [], [], [], [], []
    for seg, b in BEATS:
        gap, rr = load(seg, "beat_gap")[b], load(seg, "beat_rr_q")[b]
        fc1, lg, pr = load(seg, "beat_fc1")[b], load(seg, "beat_logits")[b], load(seg, "beat_pred")[b]
        # 3 hang PE dung chung cot trong so; moi hang da duoc preload +bias
        z4, z64 = np.zeros(4, np.int64), np.zeros(64, np.int64)
        rows = [np.concatenate([gap & 127, z4]), np.concatenate([np.minimum(gap >> 7, 127), z4]),
                np.concatenate([z64, rr])]
        T = [W1 @ r + B1 for r in rows]
        acc = rsr((T[1] << 7) + T[0] - 129 * B1, SG) + T[2]
        hh = np.clip(rsr(acc, SO), 0, 127)
        L = W2 @ hh + B2
        assert (hh == fc1).all() and (L == lg).all() and int(np.argmax(L)) == int(pr), (seg, b)
        gaps += list(gap); rrs += list(rr); hs += list(hh); logits += list(L); preds.append(int(pr))
        print(f"{seg} beat {b}: pred {int(pr)} logits {list(L)}  (RTL-style recompute == golden)")

    OUT.mkdir(parents=True, exist_ok=True)
    write("fc_w1.hex", W1.ravel(), 2)
    write("fc_b1.hex", B1, 8)
    write("fc_so.hex", SO, 2)
    write("fc_sg.hex", SG, 2)
    write("fc_w2.hex", W2.ravel(), 2)
    write("fc_b2.hex", B2, 8)
    write("fc_gap.hex", gaps, 4)
    write("fc_rr.hex", rrs, 2)
    write("fc_h.hex", hs, 2)
    write("fc_logits.hex", logits, 8)
    write("fc_pred.hex", preds, 2)
    print(f"{len(BEATS)} beats -> {OUT}/fc_*.hex")


if __name__ == "__main__":
    main()
