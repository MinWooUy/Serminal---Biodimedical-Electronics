"""Classify one heartbeat on the FPGA (layer_cmd_controller protocol): live mode, no golden needed.

    clf = BoardClassifier(LayerHost(transport))
    res = clf.classify(z_q, rr_q)        # z_q: 360 int8 (z-scored beat window), rr_q: 4 int8 RR features
    res.logits, res.pred                 # 4 int32 logits (common scale) and the class index 0..3 (N, S, V, F)

The command order is the one of host/ecg_layer_host_pc.py (ECGModelRunner), minus every read-back used for
verification: per beat the PC sends the six layers (weights are re-sent each time: the 16 KiB weight SRAM does
not hold the 23 KiB of the model) and reads only the final 17 B (4 x int32 logits + argmax byte) of FC2.

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
    def __init__(self, host: LayerHost, root: Path = ROOT):
        self.host = host
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

    # ---------------------------------------------------------------- one beat
    def classify(self, z_q, rr_q) -> BoardResult:
        z = np.asarray(z_q, dtype=np.int64).reshape(-1)
        rr = [int(v) for v in np.asarray(rr_q, dtype=np.int64).reshape(-1)]
        if z.size != 360 or len(rr) != 4:
            raise ValueError("a beat is 360 z-scored samples and 4 RR features")
        x0 = [0] * PAD + [int(v) for v in z] + [0] * PAD
        h = self.host
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
                h.write_w(0, w)
                h.swap(w=True)
                h.write_bias(0, bias)
                h.write_shift(0, shift)
                if preshift is not None:
                    h.write_shift(self.hw.cout_max, preshift)
                h.config(spec["out_len"], spec["cout"], layer.k_len, layer.window_step, bool(spec["pool"]),
                         bool(spec["gap"]), spec["pad_next"], spec["flags"] if spec["flags"] else None)
                run = h.run()
                if run.status != 0:
                    raise BoardError(f"{spec['name']}: RUN status {run.status} (writer overrun/error)")
                cycles += run.cycles
            raw = h.read_fmap(0, 17)
        except LayerLinkError as exc:
            raise BoardError(f"link error ({exc.kind}): {exc}") from exc
        if len(raw) != 17:
            raise BoardError(f"FC2 result is {len(raw)} B, expected 17")
        logits = struct.unpack("<4i", bytes(raw[:16]))
        pred = raw[16]
        if pred > 3 or pred != int(np.argmax(logits)):
            raise BoardError(f"inconsistent FC2 result: argmax byte {pred}, logits {logits}")
        s1 = h.stats
        return BoardResult(logits, pred, cycles, h._clock() - t0, s1.transactions - n0, s1.tx_bytes - tx0,
                           s1.rx_bytes - rx0)

    # ---------------------------------------------------------------- after a failure
    def recover(self) -> None:
        """Bring the link back to a clean state after BoardError: let the FPGA's receiver time out a half frame
        (100 ms of silence), drop whatever is still arriving and forget the host-side decoder buffer."""
        tr = self.host.transport
        wait = getattr(tr, "wait", None)
        (wait or time.sleep)(2 * RX_TIMEOUT_S)
        while tr.read(0.05):
            pass
        self.host.decoder.reset()
