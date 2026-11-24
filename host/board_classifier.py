"""Classify one heartbeat on the FPGA (layer_cmd_controller protocol): live mode, no golden needed.

    clf = BoardClassifier(LayerHost(transport))
    res = clf.classify(z_q, rr_q)        # z_q: 360 int8 (z-scored beat window), rr_q: 4 int8 RR features
    res.logits, res.pred                 # 4 int32 logits (common scale) and the class index 0..3 (N, S, V, F)

The command order is the one of host/ecg_layer_host_pc.py (ECGModelRunner), minus every read-back used for
verification: per beat the PC sends the six layers and reads only the final 17 B (4 x int32 logits + argmax byte)
of FC2.

Two ways to feed the weights (one beat = the same six layers either way):
  resident=False (default)  every layer re-sends its weights and biases (WRITE_W, SWAP, WRITE_BIAS ...), because one
                            16 KiB weight bank does not hold the 23 KiB of the model: ~25 kB of weights per beat.
  resident=True             the model is written ONCE (load_resident(), run automatically before the first beat and
                            again after any failure): conv0-2 into one weight bank, conv3 + fc1 + fc2 into the other,
                            all biases into the bias RAM.  Per beat the host then only sends CONFIG with w_base/b_base
                            (INFO features bit 4) and swaps the weight bank twice (before conv3, after fc2).
                            Needs a bitstream with INFO features bit 4; the answers must stay bit-identical.

Failure policy (same as LayerHost): nothing is retried automatically.  A failed beat raises BoardError; call
recover() and go on with the next beat, which starts from scratch (WRITE_X + SWAP ...), so no state of the
failed beat is reused.
"""
from __future__ import annotations

import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from host.ecg_layer_host_pc import FC1, FC2, FC1_AUX_ADDR, hardware_fit_errors, load_model  # noqa: E402
from python.link.layer_host import LayerHost, LayerLinkError  # noqa: E402
from python.link.proto_layer import BIAS_DEPTH  # noqa: E402

PAD = 3                      # zero samples on each side of the 360-sample window (conv0 pad)
RX_TIMEOUT_S = 0.1           # layer_cmd_controller drops a half-received frame after 100 ms of silence


class BoardError(RuntimeError):
    """One beat could not be classified (link error, NACK, RUN status != 0, bad result)."""


@dataclass(frozen=True)
class BoardResult:
    logits: tuple            # 4 x int32, same scale as the golden logits
    pred: int                # argmax computed by the FPGA
    cycles: int              # FPGA clock cycles of the six RUNs together
    seconds: float           # transport time of this beat (wall clock on a serial port, virtual clock in the simulator)
    transactions: int
    tx_bytes: int
    rx_bytes: int


class BoardClassifier:
    def __init__(self, host: LayerHost, root: Path = ROOT, resident: bool = False):
        self.host = host
        self.resident = resident
        self._resident_ready = False
        self._read_bank = 0                  # which weight group (0 / 1) the FPGA currently reads (resident mode)
        self.load_stats: dict | None = None  # cost of the last load_resident()
        self.model = load_model(root)
        self.hw = host.info_data or host.info()
        for i, layer in enumerate(self.model):
            problems = hardware_fit_errors(layer, self.hw, PAD * 2 + 360 if i == 0 else 0)
            if problems:
                raise BoardError(f"layer {layer.spec['name']} does not fit the FPGA reported by INFO: " + "; ".join(problems))
        # Constant operands as plain int lists, converted once.
        self._ops = [(layer, [int(v) for v in layer.weight], [int(v) for v in layer.bias],
                      [int(v) for v in layer.shift],
                      None if layer.preshift is None else [int(v) for v in layer.preshift]) for layer in self.model]
        if resident:
            self._plan_resident()

    # ---------------------------------------------------------------- resident weights
    def _plan_resident(self) -> None:
        """Where every layer's weights and biases live: weights fill weight group 0, then group 1, in layer order
        (group 0 = the bank that is read at the start of a beat); biases are packed one after another."""
        if not self.hw.supports_resident:
            raise BoardError("resident=True needs a bitstream with INFO features bit 4 (w_base/b_base); this one reports "
                             f"features 0x{self.hw.features:02X}")
        depth = self.hw.wgt_depth
        used = [0, 0]
        group = 0
        self._layout = []                    # per layer: (group, w_base, b_base)
        b_base = 0
        for layer, w, bias, _shift, _pre in self._ops:
            if used[group] + len(w) > depth:
                group += 1
                if group > 1 or len(w) > depth:
                    raise BoardError(f"the model does not fit two weight banks of {depth} B "
                                     f"(layer {layer.spec['name']}: {len(w)} B)")
            self._layout.append((group, used[group], b_base))
            used[group] += len(w)
            b_base += len(bias)
        if b_base > BIAS_DEPTH:
            raise BoardError(f"{b_base} biases do not fit the bias RAM ({BIAS_DEPTH})")
        self._bank_bytes = tuple(used)

    def load_resident(self) -> None:
        """Write the whole model once.  Works from any weight-bank parity: group 0 goes to the write bank and is
        swapped to the read side, group 1 then goes to the (new) write bank and stays there."""
        h = self.host
        s0 = h.stats
        n0, tx0, rx0 = s0.transactions, s0.tx_bytes, s0.rx_bytes
        t0 = h._clock()
        try:
            for grp in (0, 1):
                image = [0] * self._bank_bytes[grp]
                for (_layer, w, _b, _s, _p), (g, w_base, _bb) in zip(self._ops, self._layout):
                    if g == grp:
                        image[w_base:w_base + len(w)] = w
                if image:
                    h.write_w(0, image)
                if grp == 0:
                    h.swap(w=True)
            h.write_bias(0, [v for _l, _w, bias, _s, _p in self._ops for v in bias])
        except LayerLinkError as exc:
            self._resident_ready = False
            raise BoardError(f"link error while loading the resident model ({exc.kind}): {exc}") from exc
        self._read_bank = 0
        self._resident_ready = True
        s1 = h.stats
        self.load_stats = {"seconds": h._clock() - t0, "transactions": s1.transactions - n0,
                           "tx_bytes": s1.tx_bytes - tx0, "rx_bytes": s1.rx_bytes - rx0}

    def invalidate(self) -> None:
        """Forget that the model is on the chip (board reset, power cycle, a beat failed halfway): the next beat reloads it."""
        self._resident_ready = False

    # ---------------------------------------------------------------- one beat
    def classify(self, z_q, rr_q) -> BoardResult:
        z = np.asarray(z_q, dtype=np.int64).reshape(-1)
        rr = [int(v) for v in np.asarray(rr_q, dtype=np.int64).reshape(-1)]
        if z.size != 360 or len(rr) != 4:
            raise ValueError("a beat is 360 z-scored samples and 4 RR features")
        x0 = [0] * PAD + [int(v) for v in z] + [0] * PAD
        h = self.host
        if self.resident and not self._resident_ready:
            self.load_resident()                            # its cost is in self.load_stats, not in this beat's result
        s0 = h.stats
        n0, tx0, rx0 = s0.transactions, s0.tx_bytes, s0.rx_bytes
        t0 = h._clock()
        cycles = 0
        try:
            for index, (layer, w, bias, shift, preshift) in enumerate(self._ops):
                spec = layer.spec
                if index == 0:
                    h.write_x(0, x0)
                    h.swap(x=True)
                elif index == FC1:
                    h.write_x(FC1_AUX_ADDR, rr)             # RR next to the on-chip GAP; no X swap
                bases = {}
                if self.resident:
                    group, w_base, b_base = self._layout[index]
                    if group != self._read_bank:            # this layer's weights sit in the other bank: read that one
                        h.swap(w=True)
                        self._read_bank = group
                    bases = {"w_base": w_base, "b_base": b_base}
                else:
                    h.write_w(0, w)
                    h.swap(w=True)
                    h.write_bias(0, bias)
                h.write_shift(0, shift)
                if preshift is not None:
                    h.write_shift(self.hw.cout_max, preshift)
                h.config(spec["out_len"], spec["cout"], layer.k_len, layer.window_step, bool(spec["pool"]),
                         bool(spec["gap"]), spec["pad_next"], spec["flags"] if spec["flags"] else None, **bases)
                run = h.run()
                if run.status != 0:
                    raise BoardError(f"{spec['name']}: RUN status {run.status} (writer overrun/error)")
                cycles += run.cycles
            if self.resident and self._read_bank != 0:      # the next beat starts on weight group 0 again
                h.swap(w=True)
                self._read_bank = 0
            raw = h.read_fmap(0, 17)
        except LayerLinkError as exc:
            self._resident_ready = False
            raise BoardError(f"link error ({exc.kind}): {exc}") from exc
        except BoardError:
            self._resident_ready = False
            raise
        if len(raw) != 17:
            self._resident_ready = False
            raise BoardError(f"FC2 result is {len(raw)} B, expected 17")
        logits = struct.unpack("<4i", bytes(raw[:16]))
        pred = raw[16]
        if pred > 3 or pred != int(np.argmax(logits)):
            self._resident_ready = False
            raise BoardError(f"inconsistent FC2 result: argmax byte {pred}, logits {logits}")
        s1 = h.stats
        return BoardResult(logits, pred, cycles, h._clock() - t0, s1.transactions - n0, s1.tx_bytes - tx0,
                           s1.rx_bytes - rx0)

    # ---------------------------------------------------------------- after a failure
    def recover(self) -> None:
        """Bring the link back to a clean state after BoardError: let the FPGA's receiver time out a half frame
        (100 ms of silence), drop whatever is still arriving and forget the host-side decoder buffer."""
        self._resident_ready = False                        # unknown bank parity / possibly a reset board: reload next beat
        tr = self.host.transport
        wait = getattr(tr, "wait", None)
        (wait or time.sleep)(2 * RX_TIMEOUT_S)
        while tr.read(0.05):
            pass
        self.host.decoder.reset()
