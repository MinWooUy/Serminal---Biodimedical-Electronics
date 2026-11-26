"""scripts/board_compare.py headless on a short window of DS1 record 208 through the software FPGA model."""
import csv
import json

import pytest

pytest.importorskip("wfdb")
pytest.importorskip("torch")


@pytest.mark.parametrize("seg", ["detector", "annotation"])
def test_board_compare_headless(tmp_path, monkeypatch, seg):
    import python.experiment as ex
    from scripts import board_compare

    monkeypatch.setattr(ex, "RESULTS_DIR", tmp_path)
    rc = board_compare.main(["--record", "208", "--sim", "--seg", seg, "--start", "600", "--dur", "10", "--no-display"])
    assert rc == 0
    (out,) = tmp_path.iterdir()
    m = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert m["beats"] > 0 and m["answered"] == m["beats"] and m["failed"] == 0
    assert m["fpga_eq_int8_sw"] == m["answered"]                     # FPGA logits bit-exact with int_forward_q
    rows = list(csv.DictReader(open(out / "beats.csv", encoding="utf-8")))
    assert len(rows) == m["beats"]
    assert all(r["mit"] and r["float"] in "NSVF" and r["fpga"] in "NSVF" for r in rows)
    assert sum(sum(r) for r in m["agreement_fpga_rows_float_cols"]) == m["answered"]
