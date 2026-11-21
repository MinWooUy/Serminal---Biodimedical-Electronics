"""scripts/board_eval.py on a short window of DS1 record 208 through the software FPGA model (no serial port)."""
import csv
import json

import pytest

pytest.importorskip("wfdb")
pytest.importorskip("torch")


def test_board_eval_sim_window(tmp_path, monkeypatch):
    import python.experiment as ex
    from scripts import board_eval

    monkeypatch.setattr(ex, "RESULTS_DIR", tmp_path)
    rc = board_eval.main(["--record", "208", "--sim", "--start", "600", "--dur", "10", "--no-html"])
    assert rc == 0
    (out,) = tmp_path.iterdir()
    m = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    for seg in ("detector", "annotation"):
        s = m[seg]
        assert s["beats_sent"] > 0 and s["beats_failed"] == 0
        assert s["board_eq_sw"] == s["beats_sent"]                      # FPGA logits bit-exact with int_forward_q
        assert s["board"]["confusion"] == s["software_same_beats"]["confusion"]
        rows = list(csv.DictReader(open(out / f"beats_{seg}.csv", encoding="utf-8")))
        assert len(rows) == s["beats_sent"]
    # annotation R == detector R after refinement here: identical inputs are not sent twice
    assert m["annotation"]["beats_reused"] > 0
    assert m["link"]["unique_beats_classified"] < m["detector"]["beats_sent"] + m["annotation"]["beats_sent"]
