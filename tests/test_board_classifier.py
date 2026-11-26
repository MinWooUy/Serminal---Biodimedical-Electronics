"""BoardClassifier / layer simulator / board monitor, all without a board.

The FPGA is the software model in python/link/layersim (a byte-level reading of layer_cmd_controller.v); the
expected values are the golden vectors in export/golden (weights come from export/weights.h).
"""
import json

import numpy as np
import pytest

from host import ecg_layer_host_pc as eh
from host.board_classifier import BoardClassifier, BoardError
from python.link.layer_host import LayerHost
from python.link.layersim import HwConfig, LayerFpgaSim, LayerLink

ROOT = eh.REPO_ROOT
GOLD = ROOT / "export" / "golden"


class SimTransport:
    """transport interface of LayerHost over the simulator; `corrupt_write` flips one bit of the n-th write."""

    def __init__(self, corrupt_write=None):
        self.link = LayerLink(LayerFpgaSim(HwConfig(baud=921_600)))
        self.n_writes = 0
        self.corrupt_write = corrupt_write

    def write(self, data):
        self.n_writes += 1
        if self.n_writes == self.corrupt_write:
            data = bytes((data[0], data[1], data[2], data[3] ^ 0x01)) + data[4:]    # LEN low byte
        self.link.write(data)

    def read(self, timeout):
        return self.link.read(timeout)

    def wait(self, seconds):
        self.link.wait(seconds)

    def clock(self):
        return self.link.clock()

    def close(self):
        pass


def _golden(segment_index=0, beats=(0, 1, 2, 17)):
    manifest = json.loads((GOLD / "manifest.json").read_text(encoding="utf-8"))
    seg = manifest["segments"][segment_index]
    arr = {k: eh._load_golden_array(GOLD, seg, k) for k in ("beat_z_q", "beat_rr_q", "beat_logits", "beat_pred")}
    return [(arr["beat_z_q"][b], arr["beat_rr_q"][b], tuple(int(v) for v in arr["beat_logits"][b]),
             int(arr["beat_pred"][b])) for b in beats]


def test_simulated_board_returns_golden_logits_and_class():
    clf = BoardClassifier(LayerHost(SimTransport(), timeout=1.0))
    for z, rr, logits, pred in _golden():
        res = clf.classify(z, rr)
        assert res.logits == logits and res.pred == pred
        assert res.transactions == 232 and res.cycles > 300_000


def test_negative_samples_reach_the_board():
    # Regression for LayerHost.write_bytes: ECG samples and weights are mostly negative-capable int8.
    z, rr, logits, pred = _golden()[0]
    assert (np.asarray(z) < 0).any()
    assert BoardClassifier(LayerHost(SimTransport())).classify(z, rr).logits == logits


def test_failed_beat_raises_then_the_next_beat_is_correct_after_recover():
    transport = SimTransport(corrupt_write=40)                  # one damaged command in the middle of beat 1
    clf = BoardClassifier(LayerHost(transport, timeout=0.5))
    (z0, rr0, l0, _), (z1, rr1, l1, _) = _golden()[:2]
    with pytest.raises(BoardError):
        clf.classify(z0, rr0)
    clf.recover()
    assert clf.classify(z1, rr1).logits == l1                   # a new beat restarts from scratch
    assert clf.classify(z0, rr0).logits == l0


def test_bad_beat_shape_is_rejected_before_anything_is_sent():
    transport = SimTransport()
    clf = BoardClassifier(LayerHost(transport))
    sent = transport.n_writes
    with pytest.raises(ValueError):
        clf.classify(np.zeros(100), [0, 0, 0, 0])
    assert transport.n_writes == sent


def test_board_source_replay_matches_golden_digest():
    pytest.importorskip("torch")
    pytest.importorskip("wfdb")
    from python.viz.common import load_frozen
    from scripts import board_monitor as bm
    from scripts import live_monitor as lm

    fz = load_frozen()
    clf = BoardClassifier(LayerHost(SimTransport(), timeout=1.0))
    src = bm.BoardSource("rec101_93600", clf, fz, sync=True)
    r = lm.check_segment("rec101_93600", fz, source=src)
    assert lm.check_ok(r) and r["digest"]["board_ok"]
    assert src.failed == 0 and src.mismatch == 0


# ---------------------------------------------------------------- resident weights (load the model once)
class SimTransportHw(SimTransport):
    def __init__(self, hw=None, **kw):
        super().__init__(**kw)
        if hw is not None:
            self.link = LayerLink(LayerFpgaSim(hw))


def test_resident_model_matches_golden_and_sends_far_less():
    plain = BoardClassifier(LayerHost(SimTransport(), timeout=1.0))
    clf = BoardClassifier(LayerHost(SimTransport(), timeout=1.0), resident=True)
    for z, rr, logits, pred in _golden():
        ref = plain.classify(z, rr)
        res = clf.classify(z, rr)
        assert res.logits == logits == ref.logits and res.pred == pred
        assert res.cycles == ref.cycles
        assert res.tx_bytes < 0.1 * ref.tx_bytes                  # ~25 kB of weights per beat are gone
        assert res.transactions < ref.transactions // 2
    assert clf.load_stats["tx_bytes"] > 23_000                    # the model went over the wire once


def test_resident_model_is_loaded_exactly_once_for_many_beats():
    transport = SimTransport()
    clf = BoardClassifier(LayerHost(transport, timeout=1.0), resident=True)
    z, rr, logits, _ = _golden()[0]
    clf.classify(z, rr)
    sent_after_first = transport.link.sim.stats["by_cmd"].get("WRITE_W", 0)
    for _ in range(3):
        assert clf.classify(z, rr).logits == logits
    assert transport.link.sim.stats["by_cmd"]["WRITE_W"] == sent_after_first      # no WRITE_W after the load


def test_resident_reloads_after_a_failed_beat_and_stays_bit_exact():
    transport = SimTransport(corrupt_write=120)                  # one damaged command in beat 1 (the load is ~190 writes)
    clf = BoardClassifier(LayerHost(transport, timeout=0.5), resident=True)
    (z0, rr0, l0, _), (z1, rr1, l1, _) = _golden()[:2]
    transport.n_writes = 200                                      # skip the first load, damage the next write instead
    transport.corrupt_write = None
    assert clf.classify(z0, rr0).logits == l0
    transport.corrupt_write = transport.n_writes + 12
    with pytest.raises(BoardError):
        clf.classify(z1, rr1)
    assert clf._resident_ready is False
    clf.recover()
    assert clf.classify(z1, rr1).logits == l1                    # reloaded from any weight-bank parity
    assert clf.classify(z0, rr0).logits == l0


def test_resident_needs_a_bitstream_with_w_base():
    old = SimTransportHw(HwConfig(baud=921_600, features=0x0F))   # INFO bit 4 clear
    with pytest.raises(BoardError, match="bit 4"):
        BoardClassifier(LayerHost(old), resident=True)
    BoardClassifier(LayerHost(SimTransportHw(HwConfig(baud=921_600, features=0x0F))))   # default mode still works
