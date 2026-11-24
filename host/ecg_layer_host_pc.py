#!/usr/bin/env python3
"""Run the frozen ECG CNN layers on Serminal's layer_cmd_controller bitstream.

This PC runner follows Huy's host/ecg_layer_host.py layer contract:
  layers 0..3 = conv0..conv3 (conv3 also produces GAP), 4 = FC1, 5 = FC2.

It deliberately uses the project's shared LayerHost/SerialTransport implementation
instead of the legacy ecg_host.Accelerator/tile-protocol transport.  Tensor layout,
command order, CONFIG flags, FC row-combine data placement, and golden comparisons
match Huy's layer runner on branch dev.

Examples (from repository root):
  python host/ecg_layer_host.py COM5 --record rec101_93600 --beat 0
  python host/ecg_layer_host.py --validate-only
  python host/ecg_layer_host.py COM5 --first-layer 3 --last-layer 3

No automatic retransmission is performed after timeout/NACK: the FPGA command FSM
is stop-and-wait and a blind retry could repeat a non-idempotent operation.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import struct
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
CLASSES = "NSVF"

# layer_cmd_controller CONFIG flags (same bit assignment as Huy's host).
F_OUT_INT32 = 0x01
F_ARGMAX = 0x02
F_ROW_COMBINE = 0x04
F_GAP_IN = 0x08
FLAG_NAMES = {
    F_OUT_INT32: "out_int32",
    F_ARGMAX: "argmax",
    F_ROW_COMBINE: "row_combine",
    F_GAP_IN: "gap_in",
}

# (Cout, Cin, Kw, pool, gap, pad_next, out_len, flags).
# FC is implemented as Kw=1 on the same PE array as Conv1D.
LAYERS = [
    dict(name="conv0", cout=16, cin=1, kw=7, pool=1, gap=0, pad_next=2, out_len=360, flags=0),
    dict(name="conv1", cout=32, cin=16, kw=5, pool=1, gap=0, pad_next=1, out_len=180, flags=0),
    dict(name="conv2", cout=64, cin=32, kw=3, pool=1, gap=0, pad_next=1, out_len=90, flags=0),
    dict(name="conv3", cout=64, cin=64, kw=3, pool=0, gap=1, pad_next=0, out_len=45, flags=0),
    dict(name="fc1", cout=32, cin=68, kw=1, pool=0, gap=0, pad_next=0, out_len=3,
         flags=F_ROW_COMBINE | F_GAP_IN),
    dict(name="fc2", cout=4, cin=32, kw=1, pool=0, gap=0, pad_next=0, out_len=1,
         flags=F_OUT_INT32 | F_ARGMAX),
]
FC1, FC2 = 4, 5
POOL_SHAPES = [(16, 180), (32, 90), (64, 45)]  # (channels, positions), after MaxPool in layers 0..2
GAP_C = 64
FC1_AUX_ADDR = 2 * LAYERS[FC1]["cin"] + GAP_C  # 200; RR bytes at row 2 after GAP slots


@dataclass(frozen=True)
class LayerModel:
    spec: dict[str, Any]
    k_len: int
    window_step: int
    weight: np.ndarray
    bias: np.ndarray
    shift: np.ndarray
    preshift: np.ndarray | None = None


@dataclass(frozen=True)
class GoldenBeat:
    x0: np.ndarray
    layer_outputs: tuple[np.ndarray, np.ndarray, np.ndarray]
    gap: np.ndarray
    rr: np.ndarray
    fc1: np.ndarray
    logits: np.ndarray
    pred: int
    label: int


def _find_array(header: str, ctype: str, name: str) -> np.ndarray:
    # Match the same C declarations emitted by scripts/export_c.py.
    pattern = rf"static\s+const\s+{re.escape(ctype)}\s+{re.escape(name)}\s*\[[^=]+\]\s*=\s*\{{(.*?)\}};"
    match = re.search(pattern, header, flags=re.S)
    if not match:
        raise ValueError(f"{name} ({ctype}) was not found in export/weights.h")
    values = [int(v) for v in re.findall(r"-?\d+", match.group(1))]
    return np.asarray(values, dtype=np.int64)


def _macro(header: str, name: str) -> int:
    match = re.search(rf"^\s*#define\s+{re.escape(name)}\s+(\d+)\s*$", header, flags=re.M)
    if not match:
        raise ValueError(f"#define {name} is missing from export/weights.h")
    return int(match.group(1))


def load_model(root: Path = REPO_ROOT) -> list[LayerModel]:
    """Load Huy's exact weights.h layout and convert Conv weights to [Cout][K]."""
    header_path = root / "export" / "weights.h"
    header = header_path.read_text(encoding="utf-8")
    if _macro(header, "FS") != 360:
        raise ValueError("weights.h FS does not match the 360-sample hardware beat window")

    model: list[LayerModel] = []
    for index, spec in enumerate(LAYERS[:4]):
        prefix = f"CONV{index}"
        dims = (_macro(header, f"{prefix}_COUT"), _macro(header, f"{prefix}_CIN"), _macro(header, f"{prefix}_K"))
        expected_dims = (spec["cout"], spec["cin"], spec["kw"])
        if dims != expected_dims:
            raise ValueError(f"{prefix} dimensions in weights.h {dims} != host layer contract {expected_dims}")
        w = _find_array(header, "int8_t", f"{prefix}_W").reshape(spec["cout"], spec["cin"], spec["kw"])
        # RTL feeder order: K = j*Cin + i, hence [Cout][Cin][Kw] -> [Cout][Kw][Cin].
        w_for_rtl = w.transpose(0, 2, 1).reshape(spec["cout"], spec["kw"] * spec["cin"])
        bias = _find_array(header, "int32_t", f"{prefix}_B")
        shift = _find_array(header, "int8_t", f"{prefix}_SHIFT")
        if bias.size != spec["cout"] or shift.size != spec["cout"]:
            raise ValueError(f"{prefix} bias/shift length must equal Cout={spec['cout']}")
        model.append(LayerModel(dict(spec), spec["kw"] * spec["cin"], spec["cin"],
                                w_for_rtl.reshape(-1), bias, shift))

    # FC1 is three PE rows: GAP low byte, GAP high byte, and the 4 RR features.
    if (_macro(header, "FC1_OUT"), _macro(header, "FC1_GAP_IN"), _macro(header, "FC1_RR_IN")) != (32, 64, 4):
        raise ValueError("FC1 dimensions in weights.h do not match Huy's 32x(64+4) row-combine contract")
    wg = _find_array(header, "int8_t", "FC1_WG").reshape(32, GAP_C)
    wr = _find_array(header, "int8_t", "FC1_WR").reshape(32, 4)
    shift_r = _find_array(header, "int8_t", "FC1_SHIFT_R")
    if np.any(shift_r):
        raise ValueError("FC1_SHIFT_R must be all zero: row_combine adds the RR row without preshift")
    fc1_spec = dict(LAYERS[FC1])
    fc1_weight = np.concatenate((wg, wr), axis=1)
    model.append(LayerModel(fc1_spec, fc1_spec["cin"], fc1_spec["cin"], fc1_weight.reshape(-1),
                            _find_array(header, "int32_t", "FC1_B"),
                            _find_array(header, "int8_t", "FC1_SHIFT_OUT"),
                            _find_array(header, "int8_t", "FC1_SHIFT_G")))

    if _macro(header, "FC2_OUT") != 4:
        raise ValueError("FC2_OUT must be 4 for AAMI classes NSVF")
    fc2_spec = dict(LAYERS[FC2])
    fc2_weight = _find_array(header, "int8_t", "FC2_W").reshape(4, 32)
    model.append(LayerModel(fc2_spec, fc2_spec["cin"], fc2_spec["cin"], fc2_weight.reshape(-1),
                            _find_array(header, "int32_t", "FC2_B"), np.zeros(4, dtype=np.int64)))

    if len(model) != 6:
        raise AssertionError("internal layer table must have exactly six layers")
    return model


def _golden_manifest(root: Path) -> dict:
    path = root / "export" / "golden" / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"golden manifest not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _segment(manifest: dict, record: str) -> dict:
    for segment in manifest.get("segments", []):
        if segment.get("name") == record:
            return segment
    available = ", ".join(s.get("name", "?") for s in manifest.get("segments", []))
    raise KeyError(f"record {record!r} not found in golden manifest; available: {available}")


def _load_golden_array(golden_dir: Path, segment: dict, key: str) -> np.ndarray:
    meta = segment["arrays"].get(key)
    if not meta:
        raise KeyError(f"golden array {key!r} is absent for {segment['name']}")
    path = golden_dir / meta["file"]
    if not path.is_file():
        raise FileNotFoundError(f"golden vector missing: {path}")
    dtype_map = {"int8": np.dtype("i1"), "uint8": np.dtype("u1"),
                 "int16": np.dtype("<i2"), "uint16": np.dtype("<u2"),
                 "int32": np.dtype("<i4"), "uint32": np.dtype("<u4")}
    if meta["dtype"] not in dtype_map:
        raise ValueError(f"unsupported golden dtype {meta['dtype']!r} for {key}")
    data = np.fromfile(path, dtype=dtype_map[meta["dtype"]])
    shape = tuple(int(v) for v in meta["shape"])
    expected_count = math.prod(shape)
    if data.size != expected_count:
        raise ValueError(f"{path.name}: manifest expects {expected_count} elements {shape}, got {data.size}")
    return data.reshape(shape)


def load_beat(root: Path, record: str, beat: int) -> GoldenBeat:
    manifest = _golden_manifest(root)
    segment = _segment(manifest, record)
    n_beats = int(segment["n_beats"])
    if not 0 <= int(beat) < n_beats:
        raise IndexError(f"beat index {beat} out of range for {record} (0..{n_beats - 1})")
    golden_dir = root / "export" / "golden"

    z_q = _load_golden_array(golden_dir, segment, "beat_z_q")[beat].astype(np.int64)
    if z_q.shape != (360,):
        raise ValueError(f"beat_z_q must have 360 samples, got {z_q.shape}")
    x0 = np.concatenate((np.zeros(3, dtype=np.int64), z_q, np.zeros(3, dtype=np.int64)))

    padded_outputs = []
    for index, ((channels, positions), spec) in enumerate(zip(POOL_SHAPES, LAYERS[:3])):
        pooled = _load_golden_array(golden_dir, segment, f"beat_pool{index}")[beat]
        if pooled.shape != (channels, positions):
            raise ValueError(f"beat_pool{index} shape {pooled.shape}, expected {(channels, positions)}")
        # Golden is [C,T], FPGA fmap is channel-last [T,C] with pad_next zero rows.
        channel_last = pooled.astype(np.int64).T
        pad = int(spec["pad_next"])
        padded = np.pad(channel_last, ((pad, pad), (0, 0)))
        padded_outputs.append(padded.reshape(-1))

    gap = _load_golden_array(golden_dir, segment, "beat_gap")[beat].astype(np.int64)
    rr = _load_golden_array(golden_dir, segment, "beat_rr_q")[beat].astype(np.int64)
    fc1 = _load_golden_array(golden_dir, segment, "beat_fc1")[beat].astype(np.int64)
    logits = _load_golden_array(golden_dir, segment, "beat_logits")[beat].astype(np.int64)
    pred = int(_load_golden_array(golden_dir, segment, "beat_pred")[beat])
    label = int(_load_golden_array(golden_dir, segment, "beat_label")[beat])
    if gap.shape != (64,) or rr.shape != (4,) or fc1.shape != (32,) or logits.shape != (4,):
        raise ValueError("GAP/RR/FC golden vector dimensions do not match layer contract")
    return GoldenBeat(x0, tuple(padded_outputs), gap, rr, fc1, logits, pred, label)


def rshift_round(acc: np.ndarray, shift: np.ndarray | int) -> np.ndarray:
    """Huy's signed arithmetic requantisation: (acc + 2**(s-1)) >> s for s>0."""
    a = np.asarray(acc, dtype=np.int64)
    s = np.asarray(shift, dtype=np.int64)
    if s.ndim == 0:
        n = int(s)
        if n > 0:
            return (a + (np.int64(1) << (n - 1))) >> n
        if n < 0:
            return a << (-n)
        return a
    # Per-output-channel shifts: the channel axis is the last axis in FC vectors/matrices.
    if a.ndim == 1:
        if a.shape[0] != s.size:
            raise ValueError("per-channel shift length does not match accumulator channels")
        out = np.empty_like(a)
        for c, n in enumerate(s.tolist()):
            out[c] = rshift_round(a[c], int(n))
        return out
    if a.ndim >= 2:
        if a.shape[-1] != s.size:
            raise ValueError("per-channel shift length does not match accumulator channels")
        out = np.empty_like(a)
        for c, n in enumerate(s.tolist()):
            out[..., c] = rshift_round(a[..., c], int(n))
        return out
    raise ValueError("unsupported accumulator rank")


def fc_forward(model: list[LayerModel], gap: np.ndarray, rr: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Golden FC1/FC2 reference, equivalent to Huy's host and frozen quantized weights."""
    f1, f2 = model[FC1], model[FC2]
    w1 = f1.weight.reshape(f1.spec["cout"], f1.k_len)
    acc_gap = w1[:, :GAP_C] @ np.asarray(gap, dtype=np.int64)
    acc_rr = w1[:, GAP_C:] @ np.asarray(rr, dtype=np.int64)
    fc1_acc = rshift_round(acc_gap, f1.preshift) + acc_rr + f1.bias
    fc1_q = np.clip(rshift_round(fc1_acc, f1.shift), 0, 127).astype(np.int64)
    w2 = f2.weight.reshape(f2.spec["cout"], f2.k_len)
    logits = w2 @ fc1_q + f2.bias
    return fc1_q, logits.astype(np.int64), int(np.argmax(logits))


def verify_fc_golden(model: list[LayerModel], gold: GoldenBeat) -> bool:
    fc1, logits, pred = fc_forward(model, gold.gap, gold.rr)
    return bool(np.array_equal(fc1, gold.fc1) and np.array_equal(logits, gold.logits) and pred == gold.pred)


def validate_all_golden(root: Path, model: list[LayerModel]) -> tuple[int, int, list[str]]:
    """Check weights.h FC1/FC2 against every supplied golden beat; no hardware needed."""
    manifest = _golden_manifest(root)
    passed = total = 0
    failures: list[str] = []
    for segment in manifest.get("segments", []):
        name = segment["name"]
        n_beats = int(segment["n_beats"])
        # Load each array once per segment rather than re-reading per beat.
        gd = root / "export" / "golden"
        gaps = _load_golden_array(gd, segment, "beat_gap").astype(np.int64)
        rrs = _load_golden_array(gd, segment, "beat_rr_q").astype(np.int64)
        fc1s = _load_golden_array(gd, segment, "beat_fc1").astype(np.int64)
        logits_all = _load_golden_array(gd, segment, "beat_logits").astype(np.int64)
        preds = _load_golden_array(gd, segment, "beat_pred").astype(np.int64)
        for beat in range(n_beats):
            total += 1
            h, logits, pred = fc_forward(model, gaps[beat], rrs[beat])
            ok = (np.array_equal(h, fc1s[beat]) and np.array_equal(logits, logits_all[beat])
                  and pred == int(preds[beat]))
            if ok:
                passed += 1
            elif len(failures) < 12:
                failures.append(f"{name} beat {beat}: fc1_match={np.array_equal(h, fc1s[beat])}, "
                                f"logits_match={np.array_equal(logits, logits_all[beat])}, "
                                f"pred={pred} golden={int(preds[beat])}")
    return passed, total, failures


def layer_output_shape(layer: LayerModel, hw) -> tuple[int, int]:
    spec = layer.spec
    if spec["flags"] & F_OUT_INT32:
        return 1, 4 * spec["cout"] + (1 if spec["flags"] & F_ARGMAX else 0)
    positions = (1 if spec["flags"] & F_ROW_COMBINE else
                 spec["out_len"] // 2 if spec["pool"] else spec["out_len"])
    return positions + 2 * spec["pad_next"], spec["cout"]


def layer_plan(layer: LayerModel, hw) -> dict[str, int]:
    s = layer.spec
    if s["pool"]:
        if hw.supports_pool_even_rows:                    # INFO bit 5: largest even row count (7 -> 6)
            rows = (hw.rows & ~1) or 1
        else:                                             # older bitstreams: one pair of positions per tile
            rows = 2 if hw.rows >= 2 else 1
    else:
        rows = hw.rows
    rows = min(rows, s["out_len"])
    row_tiles = math.ceil(s["out_len"] / rows)
    col_tiles = math.ceil(s["cout"] / hw.cols)
    return {"rows": rows, "row_tiles": row_tiles, "col_tiles": col_tiles,
            "tiles": row_tiles * col_tiles, "macs": s["out_len"] * s["cout"] * layer.k_len}


def hardware_fit_errors(layer: LayerModel, hw, input_bytes: int = 0) -> list[str]:
    s = layer.spec
    t_out, c_out = layer_output_shape(layer, hw)
    errors = []
    if s["cout"] > hw.cout_max:
        errors.append(f"Cout {s['cout']} > COUT_MAX {hw.cout_max}")
    if layer.k_len > hw.feeder_depth:
        errors.append(f"K {layer.k_len} > FEEDER_DEPTH {hw.feeder_depth}")
    if layer.weight.size > hw.wgt_depth:
        errors.append(f"weights {layer.weight.size} B > WGT_MEM_DEPTH {hw.wgt_depth} B")
    if input_bytes > hw.ifmap_depth:
        errors.append(f"input {input_bytes} B > IFMAP_MEM_DEPTH {hw.ifmap_depth} B")
    if not s["gap"] and t_out * c_out > hw.ifmap_depth:
        errors.append(f"output {t_out * c_out} B > IFMAP_MEM_DEPTH {hw.ifmap_depth} B")
    if s["gap"] and hw.cout_max * 2 > hw.max_payload:
        errors.append(f"READ_GAP payload {hw.cout_max * 2} B > MAX_PAYLOAD {hw.max_payload} B")
    missing = s["flags"] & ~hw.features
    if missing:
        labels = ", ".join(label for bit, label in FLAG_NAMES.items() if missing & bit)
        errors.append(f"bitstream INFO features do not support: {labels}")
    if s["flags"] & F_ROW_COMBINE and hw.rows < 3:
        errors.append(f"FC1 row_combine needs at least 3 PE rows; INFO reports {hw.rows}")
    return errors


def _signed_i8_bytes(values: np.ndarray) -> bytes:
    arr = np.asarray(values, dtype=np.int64).reshape(-1)
    if arr.size and (arr.min() < -128 or arr.max() > 127):
        raise ValueError("INT8 payload contains values outside [-128,127]")
    return arr.astype(np.int8).tobytes()


def _fmap_expected_bytes(layer_index: int, gold: GoldenBeat) -> bytes:
    if layer_index < 3:
        return _signed_i8_bytes(gold.layer_outputs[layer_index])
    if layer_index == FC1:
        return _signed_i8_bytes(gold.fc1)
    if layer_index == FC2:
        return struct.pack("<4i", *(int(v) for v in gold.logits)) + bytes((gold.pred & 0xFF,))
    raise ValueError("conv3 output is GAP; use READ_GAP")


def _compare_bytes(got: bytes, expected: bytes) -> tuple[bool, int, int | None, int | None]:
    if len(got) != len(expected):
        return False, abs(len(got) - len(expected)), None, None
    mismatch = [i for i, (a, b) in enumerate(zip(got, expected)) if a != b]
    return not mismatch, len(mismatch), (mismatch[0] if mismatch else None), (mismatch[-1] if mismatch else None)


class ECGModelRunner:
    """Drive layers in Huy's required order using the shared no-retry LayerHost."""

    def __init__(self, host, hw, root: Path, verbose: bool = False):
        self.host = host
        self.hw = hw
        self.root = root
        self.verbose = verbose

    def run_layer(self, index: int, layer: LayerModel, gold: GoldenBeat, first: bool) -> tuple[bool, int]:
        spec = layer.spec
        plan = layer_plan(layer, self.hw)
        t_out, c_out = layer_output_shape(layer, self.hw)
        input_data = (gold.x0 if index == 0 else
                      gold.layer_outputs[index - 1] if index in (1, 2, 3) else
                      gold.fc1 if index == FC2 else np.empty(0, dtype=np.int64)) if first else np.empty(0, dtype=np.int64)
        problems = hardware_fit_errors(layer, self.hw, int(input_data.size) if first else 0)
        if problems:
            print(f"  ABORT {index}:{spec['name']} does not fit INFO hardware:")
            for issue in problems:
                print(f"    - {issue}")
            return False, 0

        print(f"\nLAYER {index} {spec['name']}: Cin={spec['cin']} Cout={spec['cout']} "
              f"Kw={spec['kw']} K={layer.k_len} out_len={spec['out_len']} "
              f"pool={spec['pool']} gap={spec['gap']} pad={spec['pad_next']} flags=0x{spec['flags']:02X}")
        print(f"  plan: {plan['rows']} PE rows x {self.hw.cols} cols, {plan['tiles']} tiles, "
              f"{plan['macs']:,} MACs; X/W SRAM limits {self.hw.ifmap_depth}/{self.hw.wgt_depth} B")

        stats_before = self.host.stats.transactions
        t0 = time.perf_counter()
        try:
            # A: X data. FC1 consumes GAP internally and only receives the four RR bytes.
            if index == FC1:
                self.host.write_x(FC1_AUX_ADDR, gold.rr)
                if self.verbose:
                    print(f"  A: RR {gold.rr.tolist()} -> IFMAP address {FC1_AUX_ADDR} (no X bank swap)")
            elif first:
                self.host.write_x(0, input_data)
                self.host.swap(x=True)
                if self.verbose:
                    print(f"  A: loaded {input_data.size} input bytes and swapped IFMAP bank")
            elif self.verbose:
                print("  A: input already in IFMAP SRAM from previous layer")

            # B: weights to inactive weight bank, then swap.
            self.host.write_w(0, layer.weight)
            self.host.swap(w=True)

            # C: bias and shifts. FC1's second shift vector is SG at COUT_MAX + channel.
            self.host.write_bias(0, layer.bias)
            self.host.write_shift(0, layer.shift)
            if layer.preshift is not None:
                self.host.write_shift(self.hw.cout_max, layer.preshift)

            # D/E: configure then run. Flags are omitted for legacy 11-byte conv configs.
            self.host.config(spec["out_len"], spec["cout"], layer.k_len, layer.window_step,
                             bool(spec["pool"]), bool(spec["gap"]), spec["pad_next"],
                             spec["flags"] if spec["flags"] else None)
            run_result = self.host.run()

            # F: verify hardware output against the exact golden tensor.
            if index == 3:
                got_gap = np.asarray(self.host.read_gap(), dtype=np.int64)
                expected_gap = np.asarray([int(v) & 0xFFFF for v in gold.gap], dtype=np.int64)
                mismatch = np.flatnonzero(got_gap[:GAP_C] != expected_gap)
                ok = mismatch.size == 0 and run_result.status == 0
                if run_result.status != 0:
                    print(f"  RUN status={run_result.status}: writer reported an overrun/error")
                if not ok and mismatch.size:
                    c = int(mismatch[0])
                    print(f"  GAP FAIL: {mismatch.size}/{GAP_C} channels differ; first c{c}: "
                          f"got {int(got_gap[c])}, expected {int(expected_gap[c])}")
                else:
                    print(f"  GAP PASS: {GAP_C}/{GAP_C} low-16-bit channels match golden")
            else:
                expected = _fmap_expected_bytes(index, gold)
                got = self.host.read_fmap(0, len(expected))
                ok, n_bad, first_bad, last_bad = _compare_bytes(got, expected)
                if run_result.status != 0:
                    print(f"  RUN status={run_result.status}: writer reported an overrun/error")
                    ok = False
                if ok:
                    print(f"  fmap PASS: {len(expected):,}/{len(expected):,} bytes match golden "
                          f"({t_out} x {c_out})")
                else:
                    print(f"  fmap FAIL: {n_bad:,} differing byte(s), got={len(got)} B expected={len(expected)} B")
                    if first_bad is not None:
                        print(f"    mismatch byte range: first={first_bad}, last={last_bad}; "
                              f"got=0x{got[first_bad]:02X}, expected=0x{expected[first_bad]:02X}")
            elapsed = time.perf_counter() - t0
            util = 100.0 * plan["macs"] / (run_result.cycles * self.hw.rows * self.hw.cols) if run_result.cycles else 0.0
            print(f"  RUN: {run_result.cycles:,} cycles @ {self.hw.clk_hz / 1e6:.2f} MHz "
                  f"({run_result.cycles / self.hw.clk_hz * 1e3:.3f} ms), "
                  f"{plan['tiles']} tiles, PE util estimate {util:.1f}%")
            print(f"  link: {self.host.stats.transactions - stats_before} transactions in {elapsed:.2f} s")
            print(f"  LAYER {index}: {'PASS' if ok else 'FAIL'}")
            return ok, run_result.cycles
        except Exception as exc:
            print(f"  LAYER {index}: LINK/PROTOCOL ERROR: {exc}")
            return False, 0

    def run(self, model: list[LayerModel], gold: GoldenBeat, first_layer: int, last_layer: int,
            keep_going: bool = False) -> bool:
        results = []
        t_all = time.perf_counter()
        for index in range(first_layer, last_layer + 1):
            ok, cycles = self.run_layer(index, model[index], gold, first=(index == first_layer))
            results.append((index, ok, cycles))
            if not ok and not keep_going and index < last_layer:
                print(f"\nStopped at layer {index}; later layers would consume incorrect data.")
                print(f"To isolate layer {index}, rerun with --first-layer {index} --last-layer {index}.")
                break
        wall = time.perf_counter() - t_all
        all_ok = len(results) == last_layer - first_layer + 1 and all(v[1] for v in results)
        total_cycles = sum(v[2] for v in results)
        print("\nSUMMARY")
        for index, ok, cycles in results:
            print(f"  {index}:{model[index].spec['name']:<6} {'PASS' if ok else 'FAIL':<4} "
                  f"{cycles:>10,} cycles")
        if total_cycles:
            print(f"  total compute: {total_cycles:,} cycles = {total_cycles / self.hw.clk_hz * 1e3:.3f} ms "
                  "(UART transfer excluded)")
        print(f"  elapsed wall time: {wall:.2f} s")
        print(f"  link bytes: TX={self.host.stats.tx_bytes:,}, RX={self.host.stats.rx_bytes:,}; "
              f"transactions={self.host.stats.transactions}")
        return all_ok


def _print_hw(hw) -> None:
    features = "+".join(name for bit, name in FLAG_NAMES.items() if hw.features & bit) or "none"
    print("FPGA INFO (reported by the device)")
    print(f"  PE array       : {hw.rows} x {hw.cols} ({hw.rows * hw.cols} PEs)")
    print(f"  clock / UART   : {hw.clk_hz:,} Hz / {hw.baud:,} baud")
    print(f"  IFMAP / WGT    : {hw.ifmap_depth:,} / {hw.wgt_depth:,} bytes per bank")
    print(f"  FEEDER / COUT  : {hw.feeder_depth:,} / {hw.cout_max}")
    print(f"  MAX_PAYLOAD    : {hw.max_payload} B; features: {features}; INFO v{hw.version}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("port", nargs="?", help="UART COM port, e.g. COM5 or /dev/ttyUSB0")
    ap.add_argument("--root", type=Path, default=REPO_ROOT, help="repository root (contains export/weights.h and export/golden)")
    ap.add_argument("--baud", type=int, default=921600, help="UART baud; must match the FPGA bitstream")
    ap.add_argument("--timeout", type=float, default=5.0, help="per-command response timeout in seconds")
    ap.add_argument("--record", default="rec101_93600", help="golden record/segment name")
    ap.add_argument("--beat", type=int, default=0, help="beat index inside the selected segment")
    ap.add_argument("--first-layer", type=int, default=0, choices=(0, 1, 2, 3, 5),
                    help="start at a layer using its golden previous-layer input; FC1 (4) cannot run alone")
    ap.add_argument("--last-layer", type=int, default=5, choices=range(6), help="last layer: 0..3 conv, 4 FC1, 5 FC2")
    ap.add_argument("--keep-going", action="store_true", help="continue after a failed layer")
    ap.add_argument("--validate-only", action="store_true", help="check weights.h FC1/FC2 against every golden beat; no FPGA")
    ap.add_argument("--verbose", action="store_true", help="print details about data source and layer inputs")
    ap.add_argument("--no-raw-log", action="store_true", help="do not save the UART raw-byte log")
    ap.add_argument("--log-dir", type=Path, default=None, help="raw UART log directory (default results/hardware_logs)")
    args = ap.parse_args(argv)
    if args.first_layer > args.last_layer:
        ap.error("--first-layer must be <= --last-layer")
    if args.last_layer == 4 and args.first_layer > 3:
        ap.error("FC1 (layer 4) needs the on-chip GAP produced by conv3; start at layer 0..3")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.root.resolve()
    try:
        model = load_model(root)
        if args.validate_only:
            passed, total, failures = validate_all_golden(root, model)
            print(f"FC1/FC2 frozen-weights check: {passed}/{total} golden beats match")
            for line in failures:
                print("  " + line)
            return 0 if passed == total and total > 0 else 1
        gold = load_beat(root, args.record, args.beat)
        if not verify_fc_golden(model, gold):
            print("ERROR: weights.h FC1/FC2 does not reproduce the selected golden beat; refusing to run FPGA.",
                  file=sys.stderr)
            return 2
    except Exception as exc:
        print(f"Model/golden load error: {exc}", file=sys.stderr)
        return 2

    if not args.port:
        print("A UART port is required unless --validate-only is used.", file=sys.stderr)
        return 2

    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        from python.link.layer_host import LayerHost
        from python.link.serial_link import RawLog, SerialTransport
    except ImportError as exc:
        print(f"Cannot import project UART modules: {exc}", file=sys.stderr)
        return 2

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    raw_log = None
    transport = None
    try:
        transport = SerialTransport(args.port, args.baud)
        if not args.no_raw_log:
            log_dir = args.log_dir or (root / "results" / "hardware_logs")
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / f"layer{args.first_layer}-{args.last_layer}_{args.record}_beat{args.beat:03d}_{timestamp}.raw.log"
            raw_log = RawLog(transport, log_path, header=f"{args.record} beat {args.beat}", proto="layer protocol")
            link = raw_log
            print(f"Raw UART log: {log_path}")
        else:
            link = transport
        host = LayerHost(link, timeout=args.timeout)
        hw = host.info()
        if hw.baud != args.baud:
            print(f"WARNING: FPGA INFO reports baud={hw.baud}, but host opened port at {args.baud}.")
        _print_hw(hw)
        print(f"Golden: {args.record}, beat {args.beat}; layers {args.first_layer}..{args.last_layer}")
        runner = ECGModelRunner(host, hw, root, verbose=args.verbose)
        ok = runner.run(model, gold, args.first_layer, args.last_layer, args.keep_going)
        print(f"RESULT: {'PASS' if ok else 'FAIL'}")
        return 0 if ok else 1
    except Exception as exc:
        print(f"UART/runner error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    finally:
        if raw_log is not None:
            raw_log.close()
        elif transport is not None:
            transport.close()


if __name__ == "__main__":
    raise SystemExit(main())
