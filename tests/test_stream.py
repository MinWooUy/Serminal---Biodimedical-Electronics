import numpy as np
import pytest

from python.quant.filter_int import quantize_sos, sos_filter_int
from python.quant.pipeline_int import FILTER_FRAC_BITS, FILTER_GUARD_BITS
from python.viz.stream import IntegerFilterStream


def test_filter_stream_equals_offline():
    from python.dsp.filters import design_bandpass
    sos = design_bandpass(fs=360)
    x = np.random.default_rng(0).integers(-1024, 1024, size=5000)
    off = sos_filter_int(x, quantize_sos(sos, FILTER_FRAC_BITS), FILTER_FRAC_BITS, FILTER_GUARD_BITS)
    fs = IntegerFilterStream(sos)
    assert np.array_equal([fs.push(int(v)) for v in x], off)


def _golden_available():
    from python.viz.golden import GOLDEN_DIR, golden_manifest
    try:
        return all((GOLDEN_DIR / a["file"]).exists() for s in golden_manifest()["segments"] for a in s["arrays"].values())
    except FileNotFoundError:
        return False


needs_golden = pytest.mark.skipif(not _golden_available(), reason="export/golden/*.bin not generated")


@needs_golden
@pytest.mark.parametrize("name", ["rec208_237600", "rec209_198000"])
def test_stream_bit_exact_with_golden(name):
    from python.viz.common import load_frozen
    from python.viz.golden import load_golden_segment
    from python.viz.stream import IntegerPipelineStream
    fz = load_frozen()
    g = load_golden_segment(name)
    st = IntegerPipelineStream(fz.qm, fz.pt, fz.sos, g["meta"]["adc_zero"], fz.delay, fz.half, fz.rr_lo, fz.rr_hi)
    filt, beats = [], []
    for v in g["adc"].tolist():
        for ev in st.push(v):
            (filt.append(ev[3]) if ev[0] == "sample" else beats.append(ev[1]))
    beats += [ev[1] for ev in st.flush() if ev[0] == "beat"]
    assert np.array_equal(filt, g["filt"])
    assert [b.det_index for b in beats] == g["beat_det_index"].tolist()
    assert [b.r for b in beats] == g["det_r_refined"][g["beat_det_index"]].tolist()
    assert [b.r_coarse for b in beats] == g["det_r_coarse"][g["beat_det_index"]].tolist()
    assert np.array_equal(np.stack([b.z_q for b in beats]), g["beat_z_q"])
    assert np.array_equal([b.rr_parts for b in beats], g["beat_rr_parts"])
    assert np.array_equal([b.logits for b in beats], g["beat_logits"])
    assert [b.pred for b in beats] == g["beat_pred"].tolist()
    assert all(b.emit_idx >= b.r + fz.delay + fz.half - 1 for b in beats)       # causal: window complete


@needs_golden
@pytest.mark.parametrize("name", ["rec208_237600", "rec209_198000"])
def test_offline_golden_beats_equal_stream_frames(name):
    """The offline golden BEAT frames (emit_idx / edge derived offline) equal what the stream sends."""
    from python.viz import uart
    from python.viz.common import load_frozen
    from python.viz.golden import golden_beats, load_golden_segment
    from python.viz.stream import IntegerPipelineStream
    fz = load_frozen()
    g = load_golden_segment(name, arrays=("adc",))
    st = IntegerPipelineStream(fz.qm, fz.pt, fz.sos, g["meta"]["adc_zero"], fz.delay, fz.half, fz.rr_lo, fz.rr_hi)
    evs = [ev for v in g["adc"].tolist() for ev in st.push(v)] + st.flush()
    sent = [uart.Beat(b.r, b.emit_idx, b.pred, (uart.FLAG_SEARCHBACK if b.searchback else 0)
                      | (uart.FLAG_REFINE_EDGE if b.edge else 0), b.logits, *b.rr_parts)
            for kind, *rest in evs if kind == "beat" for b in rest]
    assert sent == golden_beats(name, fz)


@needs_golden
def test_monitor_check_through_uart_clean():
    """Clean line (GATE 4): board digest == golden; everything received and bit-exact."""
    import importlib
    lm = importlib.import_module("scripts.live_monitor")
    from python.viz.common import load_frozen
    r = lm.check_segment("rec201_522000", load_frozen())
    assert lm.check_ok(r), r
    assert r["digest"]["board_ok"] and r["digest"]["host_ok"]
    assert all(v == 0 for v in r["transport"].values()), r
    assert r["beats_received"] == r["beats_golden"]


@needs_golden
def test_monitor_check_through_uart_corrupted():
    """Corrupted line (robustness, fixed seeds): if the host-recomputed digest matches golden -> pass;
    if it differs, the loss must have been recorded as transport errors (frames rejected / beats missing)."""
    import importlib
    lm = importlib.import_module("scripts.live_monitor")
    from python.viz.common import load_frozen
    r = lm.check_segment("rec201_522000", load_frozen(), corrupt=0.05, chunk_seed=0, corrupt_seed=0)
    assert r["transport"]["frames_rejected"] > 0                       # the corruption is seen
    if not r["digest"]["host_ok"]:
        assert r["transport"]["frames_rejected"] + r["transport"]["transport_error"] > 0
        assert r["transport"]["beats_not_received"] > 0 or r["computation"]["beats"] > 0
