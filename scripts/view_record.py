"""Interactive HTML viewer of one record through the FROZEN fully integer pipeline (observation only).

    python scripts/view_record.py --record 208 [--start 600 --dur 60] [--seg detector|annotation] [--model final|oof] [--cdn]

Three panels on one time axis:
  (a) 0.5-40 Hz filtered ECG (integer filter, mV); every R position coloured by the PREDICTED class,
      outline = true class (red, thick = misclassified); annotations always shown as small triangles
      (true class); FN annotation = X; FP detection = open circle; VF segments shaded.
  (b) Pan-Tompkins MWI with THR1 / THR2 of the MWI channel (detector segmentation), noise peaks and
      search-back detections.
  (c) RR (s) per beat and RR_avg8.
Hover: true / predicted label, the 4 class probabilities (softmax of the dequantised int8 logits, display
only) and the 4 RR features. DS2 records are readable only with ECG_ALLOW_DS2=1 (set by the user).
Model (--model): 'oof' (default for DS1) = the int8 model of the seed-42 CV fold that has the record in its
OUTER fold (out-of-fold, as in the integer pipeline gate, case iii); 'final' (default for DS2) = the frozen
final Lite (trained on all of DS1, so DS1 predictions are in-sample). The title states the mode.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from python.data.constants import DS1  # noqa: E402
from python.data.mitbih import REPO_ROOT, load_record  # noqa: E402
from python.viz.common import (  # noqa: E402
    CLASSES, DEFAULT_CV_RUN, FP, NOT_CLASSIFIED, Q, VF, load_frozen, oof_quant_model, record_view,
)

COLORS = {0: "#1f77b4", 1: "#ff7f0e", 2: "#2ca02c", 3: "#9467bd"}       # N, S, V, F
WRONG = "#d62728"
NAME = {0: "N", 1: "S", 2: "V", 3: "F", Q: "Q", FP: "FP (no annotation)", VF: "in VF", NOT_CLASSIFIED: "-"}


def build_figure(v: dict, rid: int, t0: float, dur: float, model_desc: str):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fs = v["fs"]
    a, b = int(t0 * fs), int(min((t0 + dur) * fs, len(v["signal_mv"])))
    t = np.arange(a, b) / fs
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=[0.5, 0.3, 0.2],
                        subplot_titles=("ECG 0.5-40 Hz (integer filter), R positions", "Pan-Tompkins MWI and thresholds",
                                        "RR intervals"))
    sig = v["signal_mv"]
    fig.add_trace(go.Scattergl(x=t, y=sig[a:b], mode="lines", line=dict(width=1, color="#444"), name="ECG (mV)",
                               hoverinfo="skip"), row=1, col=1)
    # VF shading
    for s, e in np.asarray(v["vf"]).reshape(-1, 2):
        if e >= a and s <= b:
            fig.add_vrect(x0=max(s, a) / fs, x1=min(e, b) / fs, fillcolor="#f4cccc", opacity=0.4, line_width=0,
                          annotation_text="VF", row="all", col=1)
    # annotations (always shown) and FN
    ann = np.asarray(v["ann_pos"])
    sel = (ann >= a) & (ann < b)
    top = np.nanmax(sig[a:b]) if b > a else 1.0
    for cls in range(-1, 4):
        m = sel & (v["ann_label"] == cls) & ~v["ann_missed"]
        if m.any():
            fig.add_trace(go.Scatter(x=ann[m] / fs, y=np.full(m.sum(), top * 1.15), mode="markers",
                                     marker=dict(symbol="triangle-down", size=7, color=COLORS.get(cls, "#7f7f7f")),
                                     name=f"annotation {NAME[cls]}",
                                     hovertemplate=f"annotation: {NAME[cls]}<br>t=%{{x:.3f}} s<extra></extra>"),
                          row=1, col=1)
        mfn = sel & (v["ann_label"] == cls) & v["ann_missed"]
        if mfn.any():
            fig.add_trace(go.Scatter(x=ann[mfn] / fs, y=sig[np.clip(ann[mfn], 0, len(sig) - 1)], mode="markers",
                                     marker=dict(symbol="x", size=12, color=COLORS.get(cls, "#7f7f7f"),
                                                 line=dict(width=2, color=WRONG)),
                                     name=f"FN (missed) {NAME[cls]}",
                                     hovertemplate=f"MISSED by the detector — true {NAME[cls]}<br>t=%{{x:.3f}} s<extra></extra>"),
                          row=1, col=1)
    # R positions of the analysed segmentation
    r = np.asarray(v["r"])
    rs = (r >= a) & (r < b)
    hover = []
    for i in np.flatnonzero(rs):
        p, lab = int(v["pred"][i]), int(v["label"][i])
        pr = v["probs"][i]
        rr = v["rr"][i]
        probs = " ".join(f"{c} {pr[k]:.3f}" for k, c in enumerate(CLASSES)) if not np.isnan(pr[0]) else "-"
        rrs = (f"rr_prev {rr[0]:.3f} s, rr_next {rr[1]:.3f} s<br>rr_avg8 {rr[2]:.3f} s, ratio {rr[3]:.3f}"
               if not np.isnan(rr[0]) else "-")
        hover.append(f"t={r[i] / fs:.3f} s<br>true: {NAME[lab]}<br>predicted: {NAME[p]}<br>P: {probs}<br>{rrs}"
                     + ("<br>search-back detection" if v["searchback"][i] else ""))
    hover = np.array(hover, dtype=object)
    idx = np.flatnonzero(rs)
    groups = {}
    for j, i in enumerate(idx):
        p, lab = int(v["pred"][i]), int(v["label"][i])
        if lab == FP:
            key = ("FP", p)
        elif p == NOT_CLASSIFIED:
            key = ("unclassified", lab)
        else:
            key = ("beat", p, lab >= 0 and lab != p, lab)
        groups.setdefault(key, []).append(j)
    for key, js in groups.items():
        ii = idx[js]
        x, y = r[ii] / fs, sig[np.clip(r[ii], 0, len(sig) - 1)]
        if key[0] == "FP":
            mk = dict(symbol="circle-open", size=13, color=COLORS.get(key[1], "#7f7f7f"), line=dict(width=2))
            name = f"FP detection, predicted {NAME[key[1]]}"
        elif key[0] == "unclassified":
            mk = dict(symbol="circle-open-dot", size=9, color="#7f7f7f")
            name = "R not classified (record edge)"
        else:
            _, p, wrong, lab = key
            mk = dict(symbol="circle", size=9, color=COLORS[p],
                      line=dict(width=3 if wrong else 1, color=WRONG if wrong else COLORS.get(lab, "#7f7f7f")))
            name = f"pred {NAME[p]}" + (f" (true {NAME[lab]}: WRONG)" if wrong else f" (true {NAME[lab]})")
        fig.add_trace(go.Scatter(x=x, y=y, mode="markers", marker=mk, name=name, text=hover[js],
                                 hovertemplate="%{text}<extra></extra>"), row=1, col=1)
    # (b) MWI and thresholds
    if "mwi" in v:
        fig.add_trace(go.Scattergl(x=t, y=v["mwi"][a:b], mode="lines", line=dict(width=1, color="#555"), name="MWI",
                                   hoverinfo="skip"), row=2, col=1)
        for key, dash, nm in (("thr_i1", "solid", "THR1 (MWI)"), ("thr_i2", "dot", "THR2 = THR1/2 (search-back)")):
            fig.add_trace(go.Scattergl(x=t, y=v[key][a:b], mode="lines", line=dict(width=1.2, dash=dash, color="#d62728"),
                                       name=nm, line_shape="hv"), row=2, col=1)
        pk = np.array([(p_, m_, acc) for p_, _, m_, acc in v["peaks"] if a <= p_ < b], dtype=float).reshape(-1, 3)
        noise = pk[pk[:, 2] == 0]
        if len(noise):
            fig.add_trace(go.Scatter(x=noise[:, 0] / fs, y=noise[:, 1], mode="markers",
                                     marker=dict(symbol="x-thin-open", size=8, color="#7f7f7f"), name="MWI peak -> noise"),
                          row=2, col=1)
    else:
        fig.add_annotation(text="annotation segmentation: no detector thresholds", xref="x2 domain", yref="y2 domain",
                           x=0.5, y=0.5, showarrow=False, row=2, col=1)
    # (c) RR
    ok = rs & ~np.isnan(v["rr"][:, 0])
    if ok.any():
        fig.add_trace(go.Scatter(x=r[ok] / fs, y=v["rr"][ok, 0], mode="lines+markers", marker=dict(size=4),
                                 name="rr_prev (s)"), row=3, col=1)
        fig.add_trace(go.Scatter(x=r[ok] / fs, y=v["rr"][ok, 2], mode="lines", line=dict(dash="dash"),
                                 name="rr_avg8 (s)"), row=3, col=1)
    fig.update_layout(title=f"Record {rid}, {t0:.0f}-{t0 + dur:.0f} s, segmentation: {v['seg']}, fully integer"
                            f"<br><sup>model: {model_desc}</sup>", height=900, hovermode="closest",
                      legend=dict(font=dict(size=9)))
    fig.update_xaxes(title_text="time (s)", row=3, col=1)
    fig.update_yaxes(title_text="mV", row=1, col=1)
    fig.update_yaxes(title_text="MWI", row=2, col=1)
    fig.update_yaxes(title_text="RR (s)", row=3, col=1)
    return fig


def oof_consistency(v: dict, rid: int, seed: int = 42) -> None:
    """Compare the viewer's OOF logits with the per-beat (iii) logits saved by pipeline_int_eval --save-preds."""
    f = DEFAULT_CV_RUN / "pipeline_int_preds.npz"
    if not f.exists():
        print(f"(no {f.name}: OOF consistency not checked)")
        return
    d = np.load(f)
    sel = d["rec_id"] == rid
    r_c, lg = d["r_pos"][sel], d[f"logits_iii_s{seed}"][sel]
    bi = v["beat_index"]
    r = v["r"][bi]
    j = np.clip(np.searchsorted(r, r_c), 1, len(r) - 1)
    j = np.where(np.abs(r[j - 1] - r_c) < np.abs(r[j] - r_c), j - 1, j)       # nearest classified R
    same = np.all(v["logits"][bi[j]] == lg, axis=1)
    print(f"OOF consistency with pipeline_int_eval (iii) seed {seed}: logits identical for {same.sum()}/{len(same)} "
          f"cached beats of record {rid}")
    if not same.all():
        raise SystemExit("OOF viewer differs from the integer pipeline gate")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", type=int, required=True)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--dur", type=float, default=60.0)
    ap.add_argument("--seg", choices=("detector", "annotation"), default="detector")
    ap.add_argument("--model", choices=("final", "oof"), default=None,
                    help="default: oof for DS1 records, final for DS2 records")
    ap.add_argument("--cdn", action="store_true", help="load plotly.js from the CDN (small file, needs internet)")
    ap.add_argument("--out", default=str(REPO_ROOT / "viz"))
    args = ap.parse_args()
    if not 0 < args.dur <= 600:
        raise SystemExit("--dur must be in (0, 600] s")
    rec = load_record(args.record)                 # DS2: raises unless ECG_ALLOW_DS2=1 (set by the user)
    fz = load_frozen()
    model = args.model or ("oof" if args.record in DS1 else "final")
    if model == "oof":
        if args.record not in DS1:
            raise SystemExit("--model oof exists only for DS1 records")
        qm, k = oof_quant_model(args.record)
        desc = (f"OUT-OF-FOLD — CV seed 42, fold {k} (record {args.record} in its outer fold), int8, "
                f"{DEFAULT_CV_RUN.name}")
    else:
        qm = fz.qm
        desc = f"FINAL Lite int8 ({fz.lite_dir.name})" + (
            " — trained on all DS1: predictions are IN-SAMPLE" if args.record in DS1 else "")
    v = record_view(rec, fz, args.seg, qm=qm)
    if model == "oof":
        oof_consistency(v, args.record)
    fig = build_figure(v, args.record, args.start, args.dur, desc)
    a, b = int(args.start * v["fs"]), int((args.start + args.dur) * v["fs"])
    w = (v["r"] >= a) & (v["r"] < b)
    lab, pred = v["label"][w], v["pred"][w]
    ok = (lab >= 0) & (pred >= 0)
    ann = (v["ann_pos"] >= a) & (v["ann_pos"] < b)
    print(f"record {args.record} {args.start:.0f}-{args.start + args.dur:.0f} s ({args.seg}): {w.sum()} R positions, "
          f"{ok.sum()} classified N/S/V/F beats, {(lab[ok] != pred[ok]).sum()} misclassified, "
          f"FP detections {(lab == FP).sum()}, missed annotations {(ann & v['ann_missed']).sum()}, "
          f"in VF {(lab == VF).sum()}, Q {(lab == Q).sum()}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"record_{args.record}_{int(args.start)}s_{int(args.dur)}s_{args.seg}_{model}.html"
    fig.write_html(path, include_plotlyjs="cdn" if args.cdn else True)
    print(f"saved {path} ({path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
