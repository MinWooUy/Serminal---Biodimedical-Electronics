"""Read the golden vectors written by scripts/export_c.py (export/golden/*.bin + manifest, SHA-256 checked)."""

import hashlib
import json

import numpy as np

from python.data.mitbih import REPO_ROOT

GOLDEN_DIR = REPO_ROOT / "export" / "golden"


def golden_manifest() -> dict:
    return json.loads((REPO_ROOT / "export" / "golden_manifest.json").read_text(encoding="utf-8"))


def load_golden_segment(name: str, arrays=None, verify: bool = True) -> dict:
    man = golden_manifest()
    seg = next((s for s in man["segments"] if s["name"] == name), None)
    if seg is None:
        raise KeyError(f"golden segment {name!r} not in export/golden_manifest.json")
    out = {"meta": {k: v for k, v in seg.items() if k != "arrays"}}
    for key, a in seg["arrays"].items():
        if arrays is not None and key not in arrays:
            continue
        f = GOLDEN_DIR / a["file"]
        if not f.exists():
            raise FileNotFoundError(f"{f} missing: regenerate with scripts/export_c.py")
        raw = f.read_bytes()
        if verify and hashlib.sha256(raw).hexdigest() != a["sha256"]:
            raise RuntimeError(f"{f} does not match the golden manifest")
        out[key] = np.frombuffer(raw, dtype="<" + np.dtype(a["dtype"]).str[1:]).reshape(a["shape"]).astype(np.int64)
    return out


def golden_beats(name: str, fz) -> list:
    """The BEAT frames the device must send for a golden segment, built OFFLINE (independent of the
    streaming implementation): R, search-back flag, RR parts, logits and prediction from the golden .bin
    files; emit_idx and the refinement-edge flag derived offline:
      emit_idx = max(emission sample of the next detection, R + delay + half - 1)  (next R known and window
                 complete), with the emission samples of the shared DecisionEngine (frontend_offline);
      edge     = the integer abs refinement argmax hit the border of its +-18-sample window (coarse R kept)."""
    from python.dsp.pan_tompkins import DecisionEngine, frontend_offline
    from python.viz import uart
    from python.viz.stream import REFINE_HALF
    g = load_golden_segment(name)
    st = frontend_offline(g["adc"], g["meta"]["adc_zero"], fz.pt)
    eng = DecisionEngine(fz.pt)
    dets = []
    for hp, d, mwi in zip(st["hp"].tolist(), st["d"].tolist(), st["mwi"].tolist()):
        dets += eng.process(hp, d, mwi)
    dets += eng.flush()
    dets.sort(key=lambda x: x.r_coarse)
    if [x.r_coarse for x in dets] != g["det_r_coarse"].tolist():
        raise RuntimeError(f"{name}: offline detections differ from the golden r_coarse")
    a = np.abs(g["filt"])
    out = []
    for b, j in enumerate(g["beat_det_index"].tolist()):
        r = int(g["det_r_refined"][j])
        c = int(g["det_r_coarse"][j]) + fz.delay
        lo, hi = max(c - REFINE_HALF, 0), min(c + REFINE_HALF, len(a) - 1)
        k = int(np.argmax(a[lo:hi + 1]))
        edge = hi <= lo or k == 0 or k == hi - lo
        flags = (uart.FLAG_SEARCHBACK if g["det_searchback"][j] else 0) | (uart.FLAG_REFINE_EDGE if edge else 0)
        emit = max(dets[j + 1].emitted_at, r + fz.delay + fz.half - 1)
        out.append(uart.Beat(r, emit, int(g["beat_pred"][b]), flags, tuple(int(v) for v in g["beat_logits"][b]),
                             *(int(v) for v in g["beat_rr_parts"][b])))
    return out


def golden_digest(name: str, fz) -> tuple[int, int]:
    """(beat_digest, n_beats) the device must report at the end of the golden segment."""
    from python.viz import uart
    beats = golden_beats(name, fz)
    return uart.beat_digest(beats), len(beats)
