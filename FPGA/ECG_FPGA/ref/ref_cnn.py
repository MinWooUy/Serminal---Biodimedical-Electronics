"""Minimal reference of the int8 ECG CNN for the FPGA team -- numpy + standard library only.

Reads the exported files (ecg_params.vh, mem/*.hex) -- NOT the Python project -- and computes every layer with
the integer arithmetic of the FPGA. It must equal the project's golden reference (int_forward_trace) bit for bit.

    python ref/ref_cnn.py --rtl export/rtl        # check golden/*.hex and all 1 306 beats of golden/uart_in.bin
    python ref/ref_cnn.py --rtl .                 # inside the hand-off package (params/ecg_params.vh)

Arithmetic (also written at the top of ecg_params.vh):
  requantise rq(acc, s) = (acc + (1 << (s - 1))) >> s  (arithmetic shift, s > 0); acc (s == 0); acc << -s (s < 0)
  conv:  acc = sum W[co][ci][k] * x[ci][t + k - K//2] (zero padding) + B[co];  y = clamp(rq(acc, SHIFT[co]), 0, 127)
  pool:  max of pairs (after conv 0..2);  GAP: sum over the 45 positions (int32)
  FC1:   acc = rq(WG @ gap, SHIFT_G) + rq(WR @ rr_q, SHIFT_R) + B;  h = clamp(rq(acc, SHIFT_OUT), 0, 127)
  FC2:   logits = W @ h + B (int32);  pred = argmax (ties -> lowest index)
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np

INT32_MAX = 2**31 - 1


def parse_vh(path: Path) -> dict:
    text = Path(path).read_text(encoding="ascii")
    p = {k: int(v) for k, v in re.findall(r"(\w+) = (\d+)[,;]", text)}
    p["WEIGHTS_CRC32"] = int(re.search(r"WEIGHTS_CRC32 = 32'h([0-9A-Fa-f]{8})", text).group(1), 16)
    return p


def read_hex(path: Path, bits: int, signed: bool = True) -> np.ndarray:
    a = np.array([int(t, 16) for t in Path(path).read_text(encoding="ascii").split()], dtype=np.int64)
    return np.where(a >= 1 << (bits - 1), a - (1 << bits), a) if signed else a


def load_params(rtl: Path) -> tuple[dict, dict]:
    rtl = Path(rtl)
    vh_path = rtl / "ecg_params.vh"
    vh = parse_vh(vh_path if vh_path.exists() else rtl / "params" / "ecg_params.vh")     # export dir or hand-off package
    mem = {}
    for f in sorted((rtl / "mem").glob("*.hex")):
        name = f.stem.split("_", 1)[1]
        mem[name] = read_hex(f, vh[f"{name.upper()}_WIDTH"])
        assert mem[name].size == vh[f"{name.upper()}_DEPTH"], name
    for i in range(4):
        mem[f"conv{i}_w"] = mem[f"conv{i}_w"].reshape(vh[f"CONV{i}_COUT"], vh[f"CONV{i}_CIN"], vh[f"CONV{i}_K"])
    mem["fc1_wg"] = mem["fc1_wg"].reshape(vh["FC1_OUT"], vh["FC1_GAP_IN"])
    mem["fc1_wr"] = mem["fc1_wr"].reshape(vh["FC1_OUT"], vh["FC1_RR_IN"])
    mem["fc2_w"] = mem["fc2_w"].reshape(vh["FC2_OUT"], vh["FC2_IN"])
    return vh, mem


def rq(acc: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Requantise with a per-channel shift s (broadcast over axis 1)."""
    acc = np.asarray(acc, dtype=np.int64)
    s = np.asarray(s, dtype=np.int64).reshape((1, -1) + (1,) * (acc.ndim - 2))
    pos = np.maximum(s, 1)
    up = (acc + (np.int64(1) << (pos - 1))) >> pos
    return np.where(s > 0, up, np.where(s == 0, acc, acc << np.maximum(-s, 0)))


def _check(acc: np.ndarray, where: str) -> None:
    if acc.size and int(np.abs(acc).max()) > INT32_MAX:
        raise OverflowError(f"{where}: accumulator exceeds int32")


def conv(x: np.ndarray, w: np.ndarray, b: np.ndarray, shift: np.ndarray) -> np.ndarray:
    k = w.shape[2]
    xp = np.pad(x, ((0, 0), (0, 0), (k // 2, k // 2)))
    win = np.lib.stride_tricks.sliding_window_view(xp, k, axis=2)            # (B, Cin, L, K)
    acc = np.einsum("bclk,ock->bol", win, w) + b[None, :, None]
    _check(acc, "conv")
    return np.clip(rq(acc, shift), 0, 127)


def forward(params: tuple[dict, dict], z_q: np.ndarray, rr_q: np.ndarray) -> dict:
    vh, m = params
    h = np.asarray(z_q, dtype=np.int64)[:, None, :]
    out = {}
    for i in range(4):
        h = conv(h, m[f"conv{i}_w"], m[f"conv{i}_b"], m[f"conv{i}_shift"])
        out[f"conv{i}"] = h
        if vh[f"CONV{i}_POOL"]:
            n = h.shape[2] // 2 * 2
            h = np.maximum(h[:, :, 0:n:2], h[:, :, 1:n:2])
            out[f"pool{i}"] = h
    gap = h.sum(axis=2)
    out["gap"] = gap
    acc_g = gap @ m["fc1_wg"].T
    acc_r = np.asarray(rr_q, dtype=np.int64) @ m["fc1_wr"].T
    _check(acc_g, "fc1 gap")
    _check(acc_r, "fc1 rr")
    acc = rq(acc_g, m["fc1_shift_g"]) + rq(acc_r, m["fc1_shift_r"]) + m["fc1_b"][None, :]
    _check(acc, "fc1")
    fc1 = np.clip(rq(acc, m["fc1_shift_out"]), 0, 127)
    out["fc1"] = fc1
    logits = fc1 @ m["fc2_w"].T + m["fc2_b"][None, :]
    _check(logits, "fc2")
    out["logits"] = logits
    out["pred"] = logits.argmax(axis=1)
    return out


def read_beat_in_frames(data: bytes) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """uart_in.bin: back-to-back BEAT_IN frames (A5 5A 10 70 01 | beat_id u32 | z_q 360 | rr_q 4 | crc u16)."""
    size = 5 + 368 + 2
    if len(data) % size:
        raise ValueError("uart_in.bin is not a whole number of BEAT_IN frames")
    f = np.frombuffer(data, dtype=np.uint8).reshape(-1, size)
    if not (f[:, :5] == np.array([0xA5, 0x5A, 0x10, 0x70, 0x01], dtype=np.uint8)).all():
        raise ValueError("unexpected frame header in uart_in.bin")
    beat_id = f[:, 5:9].copy().view("<u4").ravel().astype(np.int64)
    z_q = f[:, 9:369].view(np.int8).astype(np.int64)
    rr_q = f[:, 369:373].view(np.int8).astype(np.int64)
    return beat_id, z_q, rr_q


def check(rtl: Path) -> list[str]:
    """Compare with the golden files; returns the list of mismatches (empty = bit-exact)."""
    rtl = Path(rtl)
    params = load_params(rtl)
    g = rtl / "golden"
    bad = []
    sel = read_hex(g / "beats_sel.hex", 32)
    zq = read_hex(g / "zq.hex", 8).reshape(len(sel), -1)
    rrq = read_hex(g / "rrq.hex", 8).reshape(len(sel), -1)
    out = forward(params, zq, rrq)
    for key in ("conv0", "pool0", "conv1", "pool1", "conv2", "pool2", "conv3", "fc1"):
        if not np.array_equal(read_hex(g / f"{key}.hex", 8), out[key].ravel()):
            bad.append(key)
    if not np.array_equal(read_hex(g / "gap.hex", 32), out["gap"].ravel()):
        bad.append("gap")
    beat_id, z_all, rr_all = read_beat_in_frames((g / "uart_in.bin").read_bytes())
    res = read_hex(g / "result_expected.hex", 32).reshape(-1, 6)
    full = forward(params, z_all, rr_all)
    if not np.array_equal(res[:, 0], beat_id):
        bad.append("result_expected beat_id")
    if not np.array_equal(res[:, 1:5], full["logits"]):
        bad.append("result_expected logits")
    if not np.array_equal(res[:, 5], full["pred"]):
        bad.append("result_expected pred")
    return bad


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rtl", default="export/rtl", help="directory with ecg_params.vh, mem/, golden/")
    args = ap.parse_args()
    bad = check(Path(args.rtl))
    print("ref_cnn: BIT-EXACT with golden" if not bad else f"ref_cnn: MISMATCH in {bad}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
