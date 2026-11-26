import numpy as np
import pytest

from tests.test_stream import needs_golden


@needs_golden
def test_trace_beat_smoke(tmp_path, capsys):
    """One golden beat: the recomputed integer trace equals golden, every stage is printed, the PNG is written."""
    from python.viz.common import load_frozen
    from python.viz.golden import load_golden_segment
    from scripts import trace_beat as tb
    fz = load_frozen()
    name, k = "rec208_237600", 12
    g = load_golden_segment(name)
    assert tb.recompute_matches(fz.qm, g, k)
    rows = tb.stage_table(fz.qm, g, k)
    names = [r["name"] for r in rows]
    assert names == ["window", "z_q", "conv0", "pool0", "conv1", "pool1", "conv2", "pool2", "conv3", "gap",
                     "rr_parts", "rr_q", "fc1", "logits"]
    shapes = {r["name"]: np.asarray(r["arr"]).shape for r in rows}
    assert shapes["conv0"] == (16, 360) and shapes["pool2"] == (64, 45) and shapes["gap"] == (64,)
    assert shapes["logits"] == (4,)
    tb.print_trace(name, k, fz.qm, g, rows, True)
    out = capsys.readouterr().out
    assert "recomputed int_forward_trace == golden: YES" in out and "argmax" in out
    path = tmp_path / "trace.png"
    tb.draw(name, k, fz.qm, g, path)
    assert path.exists() and path.stat().st_size > 10_000


@needs_golden
def test_trace_beat_first_of_finds_labels():
    from scripts import trace_beat as tb
    from python.viz.golden import load_golden_segment
    for c, cls in enumerate("NSVF"):
        name, k = tb.first_of(cls)
        assert load_golden_segment(name, arrays=("beat_label",))["beat_label"][k] == c


def test_trace_beat_requires_a_beat():
    from scripts import trace_beat as tb
    import sys
    argv = sys.argv
    sys.argv = ["trace_beat.py"]
    try:
        with pytest.raises(SystemExit):
            tb.main()
    finally:
        sys.argv = argv
