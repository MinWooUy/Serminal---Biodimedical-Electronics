"""Parameter layout shared by the RTL export, the FPGA simulator and the host.

The ORDER of PARAM_ARRAYS is part of the contract with the FPGA team: the files in mem/ are numbered in this
order, and weights_crc32 = CRC-32/IEEE over the raw little-endian bytes of every array in this order
(int8 -> 1 byte, int32 -> 4 bytes per value). Index order inside an array: conv weights co -> ci -> k,
FC weights out -> in, vectors by channel / row.
"""

from dataclasses import dataclass

import numpy as np

from python.link.proto_v2 import crc32
from python.quant.int8 import QuantModel


@dataclass(frozen=True)
class ParamArray:
    index: int
    name: str
    bits: int                  # 8 or 32 (signed, two's complement)
    values: np.ndarray         # int64, C order

    @property
    def file_stem(self) -> str:
        return f"{self.index:02d}_{self.name}"

    def raw(self) -> bytes:
        dt = "<i1" if self.bits == 8 else "<i4"
        return self.values.astype(dt).tobytes(order="C")


def param_arrays(qm: QuantModel) -> list[ParamArray]:
    items = []
    for i, c in enumerate(qm.convs):
        items += [(f"conv{i}_w", 8, c["w_q"]), (f"conv{i}_b", 32, c["b_q"]),
                  (f"conv{i}_shift", 8, c["k"] + c["a_in"] - c["a_out"])]
    f1, f2 = qm.fc1, qm.fc2
    items += [("fc1_wg", 8, f1["wg_q"]), ("fc1_wr", 8, f1["wr_q"]), ("fc1_b", 32, f1["b_q"]),
              ("fc1_shift_g", 8, f1["Eg"] - f1["E"]), ("fc1_shift_r", 8, f1["Er"] - f1["E"]),
              ("fc1_shift_out", 8, f1["E"] - f1["a_out"]),
              ("fc2_w", 8, f2["w_q"]), ("fc2_b", 32, f2["b_q"])]
    out = []
    for k, (name, bits, v) in enumerate(items):
        a = np.ascontiguousarray(np.asarray(v, dtype=np.int64))
        lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
        if a.size and (a.min() < lo or a.max() > hi):
            raise OverflowError(f"{name}: values outside int{bits}")
        out.append(ParamArray(k, name, bits, a))
    return out


def weights_blob(qm: QuantModel) -> bytes:
    return b"".join(p.raw() for p in param_arrays(qm))


def weights_crc32(qm: QuantModel) -> int:
    return crc32(weights_blob(qm))


# ---------------------------------------------------------------- $readmemh text (one value per line)

def to_hex_lines(values: np.ndarray, bits: int) -> str:
    mask, width = (1 << bits) - 1, bits // 4
    return "".join(f"{int(v) & mask:0{width}X}\n" for v in np.asarray(values, dtype=np.int64).ravel())


def from_hex_lines(text: str, bits: int, signed: bool = True) -> np.ndarray:
    vals = [int(t, 16) for t in text.split() if t and not t.startswith("//")]
    a = np.array(vals, dtype=np.int64)
    if signed:
        a = np.where(a >= 1 << (bits - 1), a - (1 << bits), a)
    return a
