from pathlib import Path
import hashlib
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from host.ecg_layer_host_pc import (  # noqa: E402
    REPO_ROOT, FC1, FC2, _fmap_expected_bytes, hardware_fit_errors,
    layer_output_shape, layer_plan, load_beat, load_model, validate_all_golden,
)
from python.link.proto_layer import HardwareInfo  # noqa: E402


def _hw():
    return HardwareInfo(
        version=2, rows=7, cols=7, features=0x0F,
        clk_hz=50_000_000, baud=921_600,
        ifmap_depth=4096, wgt_depth=16384, feeder_depth=1024,
        cout_max=64, max_payload=128,
    )


def test_weights_h_matches_all_golden_fc_beats():
    model = load_model(REPO_ROOT)
    passed, total, failures = validate_all_golden(REPO_ROOT, model)
    assert total == 1306
    assert passed == total, failures


def test_huy_layer_contract_shapes_and_weights():
    model = load_model(REPO_ROOT)
    assert len(model) == 6
    assert [m.spec["name"] for m in model] == ["conv0", "conv1", "conv2", "conv3", "fc1", "fc2"]
    assert [m.weight.size for m in model] == [112, 2560, 6144, 12288, 2176, 128]
    assert [m.k_len for m in model] == [7, 80, 96, 192, 68, 32]
    assert np.all(model[FC1].preshift >= 0)
    assert np.all(model[FC2].shift == 0)


def test_golden_beat_channel_last_padding_and_fc_bytes():
    model = load_model(REPO_ROOT)
    gold = load_beat(REPO_ROOT, "rec101_93600", 0)
    assert gold.x0.shape == (366,)
    assert [x.size for x in gold.layer_outputs] == [184 * 16, 92 * 32, 47 * 64]
    # First/last pad rows are zero, and inner data are channel-last.
    assert np.all(gold.layer_outputs[0][: 2 * 16] == 0)
    assert np.all(gold.layer_outputs[0][-2 * 16 :] == 0)
    assert len(_fmap_expected_bytes(FC1, gold)) == 32
    assert len(_fmap_expected_bytes(FC2, gold)) == 17


def test_7x7_de10_lite_info_fits_all_six_layers():
    model = load_model(REPO_ROOT)
    hw = _hw()
    gold = load_beat(REPO_ROOT, "rec101_93600", 0)
    inputs = [gold.x0, gold.layer_outputs[0], gold.layer_outputs[1], gold.layer_outputs[2],
              np.empty(0, dtype=np.int64), gold.fc1]
    for index, layer in enumerate(model):
        assert hardware_fit_errors(layer, hw, inputs[index].size) == []
    assert [layer_output_shape(layer, hw) for layer in model] == [
        (184, 16), (92, 32), (47, 64), (45, 64), (1, 32), (1, 17)
    ]
    # Pool layers use two PE rows per tile, FC1 needs three rows, FC2 uses one.
    assert layer_plan(model[0], hw)["rows"] == 2
    assert layer_plan(model[FC1], hw)["rows"] == 3
    assert layer_plan(model[FC2], hw)["rows"] == 1


def test_layer_plan_uses_even_rows_when_the_bitstream_reports_it():
    # INFO features bit 5 (0x20): pool layers use 6 of the 7 PE rows; conv3 (no pool) all 7, as the board's 367 091-cycle run.
    model = load_model(REPO_ROOT)
    hw = HardwareInfo(version=2, rows=7, cols=7, features=0x3F, clk_hz=50_000_000, baud=1_500_000,
                      ifmap_depth=4096, wgt_depth=16384, feeder_depth=1024, cout_max=64, max_payload=128)
    plans = [layer_plan(layer, hw) for layer in model]
    assert [p["rows"] for p in plans] == [6, 6, 6, 7, 3, 1]
    assert [(p["row_tiles"], p["col_tiles"]) for p in plans] == [(60, 3), (30, 5), (15, 10), (7, 10), (1, 5), (1, 1)]
    assert sum(p["macs"] for p in plans[:4]) == 1_607_040          # conv0..conv3


def test_layer_runner_emits_huy_command_order_without_board():
    from types import SimpleNamespace

    from host.ecg_layer_host_pc import ECGModelRunner, _fmap_expected_bytes
    from python.link.layer_host import RunResult

    class FakeHost:
        def __init__(self, gold):
            self.gold = gold
            self.calls = []
            self.current_layer = -1
            self.stats = SimpleNamespace(transactions=0, tx_bytes=0, rx_bytes=0)

        def _call(self, name, *args):
            self.calls.append((name, *args))
            self.stats.transactions += 1

        def write_x(self, addr, data): self._call("WRITE_X", addr, tuple(int(x) for x in data))
        def write_w(self, addr, data): self._call("WRITE_W", addr, tuple(int(x) for x in data))
        def write_bias(self, addr, data): self._call("WRITE_BIAS", addr, tuple(int(x) for x in data))
        def write_shift(self, addr, data): self._call("WRITE_SHIFT", addr, tuple(int(x) for x in data))
        def swap(self, x=False, w=False): self._call("SWAP", bool(x), bool(w))
        def config(self, *args):
            self.current_layer += 1
            self._call("CONFIG", *args)
        def run(self):
            self._call("RUN")
            return RunResult(0, 1000 + self.current_layer)
        def read_fmap(self, offset, length):
            self._call("READ_FMAP", offset, length)
            return _fmap_expected_bytes(self.current_layer, self.gold)
        def read_gap(self):
            self._call("READ_GAP")
            return tuple(int(v) & 0xFFFF for v in self.gold.gap) + (0,) * (64 - len(self.gold.gap))

    model = load_model(REPO_ROOT)
    gold = load_beat(REPO_ROOT, "rec101_93600", 0)
    host = FakeHost(gold)
    runner = ECGModelRunner(host, _hw(), REPO_ROOT)
    assert runner.run(model, gold, first_layer=0, last_layer=5) is True

    names = [c[0] for c in host.calls]
    # Layer 0 includes initial X load/swap, all layers load W, bias/shift, configure and run.
    assert names[:8] == ["WRITE_X", "SWAP", "WRITE_W", "SWAP", "WRITE_BIAS", "WRITE_SHIFT", "CONFIG", "RUN"]
    assert names.count("CONFIG") == 6
    assert names.count("RUN") == 6
    # FC1's RR input is written at byte address 200 without an X bank swap.
    fc1_cfg = next(i for i, call in enumerate(host.calls) if call[0] == "CONFIG" and call[1] == 3)
    assert host.calls[fc1_cfg - 1][0] == "WRITE_SHIFT"  # SG is the last load before CONFIG.
    # The special RR write is present and targets the Huy-defined auxiliary region.
    assert ("WRITE_X", 200, tuple(int(x) for x in gold.rr)) in host.calls


def test_all_golden_files_match_manifest_sha256():
    golden_dir = REPO_ROOT / "export" / "golden"
    manifest = json.loads((golden_dir / "manifest.json").read_text(encoding="utf-8"))
    count = 0
    for segment in manifest["segments"]:
        for meta in segment["arrays"].values():
            data = (golden_dir / meta["file"]).read_bytes()
            assert hashlib.sha256(data).hexdigest() == meta["sha256"], (segment["name"], meta["file"])
            count += 1
    assert count == 200
