"""Trace ONE real beat through every stage of the frozen fully integer pipeline (learning aid, read-only).

    python scripts/trace_beat.py --golden rec208_237600 --beat 12 [--out docs/img] [--first-of N|S|V|F]

Reads the beat from the golden vectors (export/golden/*.bin, SHA-256 checked), recomputes the integer CNN
from its z_q / rr_q with int_forward_trace and asserts that every intermediate tensor equals the golden one.
Prints, per stage: shape, integer range, scale (value = int x 2^-exp), min / max and the first values.
Saves a PNG: input window (filtered, mV) and z_q, a heatmap of each conv block output (channels x positions,
after requantise / ReLU, before pooling), the 64 GAP sums as a column, and the 4 logits.
--first-of C picks the first beat whose golden label is class C over all golden segments (manifest order).
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from python.data.mitbih import REPO_ROOT  # noqa: E402
from python.quant.int8 import int_forward_trace  # noqa: E402
from python.quant.pipeline_int import FILTER_GUARD_BITS  # noqa: E402
from python.viz.common import CLASSES, load_frozen  # noqa: E402
from python.viz.golden import golden_manifest, load_golden_segment  # noqa: E402

LABEL_NAME = {0: "N", 1: "S", 2: "V", 3: "F", -1: "Q", -2: "FP (unmatched detection)"}
ADC_GAIN = 200                                   # MIT-BIH: 200 ADC counts per mV


def stage_table(qm, g: dict, k: int) -> list[dict]:
    """Every stage of beat k: golden tensor, its scale exponent and a short meaning."""
    det = int(g["beat_det_index"][k])
    f1, f2 = qm.fc1, qm.fc2
    rows = [
        dict(name="window", arr=g["beat_window"][k], exp=FILTER_GUARD_BITS,
             meaning="filt[R+2-180 : R+2+180], ADC counts x 2^12 (int32)"),
        dict(name="z_q", arr=g["beat_z_q"][k], exp=qm.input_exp, meaning="integer z-score, clip +-127 (= +-8 sd)"),
    ]
    for i, c in enumerate(qm.convs):
        rows.append(dict(name=f"conv{i}", arr=g[f"beat_conv{i}"][k], exp=c["a_out"],
                         meaning=f"block {i + 1}: conv k={c['w_q'].shape[2]} + folded BN, ReLU, clamp 0..127"))
        if c["pool"]:
            rows.append(dict(name=f"pool{i}", arr=g[f"beat_pool{i}"][k], exp=c["a_out"], meaning="MaxPool(2)"))
    rows += [
        dict(name="gap", arr=g["beat_gap"][k], exp=f1["a_gap"], meaning="sum over 45 positions (1/45 folded in FC1)"),
        dict(name="rr_parts", arr=g["beat_rr_parts"][k], exp=0,
             meaning="rr_prev, rr_next, sum8, n8 (samples, RR clipped 72..1080)"),
        dict(name="rr_q", arr=g["beat_rr_q"][k], exp=None,
             meaning=f"int8 RR features, exponents {qm.rr_exp.tolist()} (rr_prev, rr_next, rr_avg, ratio)"),
        dict(name="fc1", arr=g["beat_fc1"][k], exp=f1["a_out"], meaning="FC1 + ReLU, clamp 0..127"),
        dict(name="logits", arr=g["beat_logits"][k], exp=f2["k"] + f2["a_in"], meaning="FC2, int32, common scale"),
    ]
    for r in rows:
        r["det"] = det
    return rows


def recompute_matches(qm, g: dict, k: int) -> bool:
    tr = int_forward_trace(qm, g["beat_z_q"][k:k + 1], g["beat_rr_q"][k:k + 1])
    keys = [f"conv{i}" for i in range(4)] + [f"pool{i}" for i in range(3)] + ["gap", "fc1", "logits"]
    return all(np.array_equal(tr[key][0], g[f"beat_{key}"][k]) for key in keys)


def print_trace(name: str, k: int, qm, g: dict, rows: list[dict], ok: bool) -> None:
    det = rows[0]["det"]
    r = int(g["det_r_refined"][det])
    lab, pred = int(g["beat_label"][k]), int(g["beat_pred"][k])
    print(f"golden segment {name}, beat {k} (detection {det}), R = sample {r} of the segment "
          f"(record {g['meta']['record']}, sample {g['meta']['start_sample'] + r})")
    print(f"label {LABEL_NAME[lab]}, predicted {CLASSES[pred]}; recomputed int_forward_trace == golden: "
          f"{'YES' if ok else 'NO'}")
    print(f"{'stage':<9}{'shape':<14}{'scale':<10}{'min':>12}{'max':>12}  first values  | meaning")
    for row in rows:
        a = np.asarray(row["arr"])
        flat = a.ravel()
        scale = "-" if row["exp"] is None else f"2^-{row['exp']}"
        first = " ".join(str(int(v)) for v in flat[:6])
        print(f"{row['name']:<9}{str(tuple(a.shape)):<14}{scale:<10}{int(flat.min()):>12}{int(flat.max()):>12}  "
              f"{first}  | {row['meaning']}")
    lg = np.asarray(g["beat_logits"][k], dtype=np.float64) / 2.0 ** (qm.fc2["k"] + qm.fc2["a_in"])
    p = np.exp(lg - lg.max())
    p /= p.sum()
    print("logits / 2^%d = %s -> softmax (display only) %s -> argmax %s"
          % (qm.fc2["k"] + qm.fc2["a_in"], np.round(lg, 3).tolist(), np.round(p, 3).tolist(), CLASSES[pred]))


def draw(name: str, k: int, qm, g: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    lab, pred = int(g["beat_label"][k]), int(g["beat_pred"][k])
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 7, figure=fig, height_ratios=[1.0, 1.25, 1.25], width_ratios=[1, 1, 1, 1, 0.25, 0.3, 1.0],
                  hspace=0.6, wspace=0.5)
    t = (np.arange(360) - 180) / 360 * 1000
    ax = fig.add_subplot(gs[0, :4])
    ax.plot(t, g["beat_window"][k] / 2.0**FILTER_GUARD_BITS / ADC_GAIN, color="#333", lw=1.2)
    ax.set_ylabel("filtered ECG (mV)")
    ax.set_xlabel("ms from R")
    ax2 = ax.twinx()
    ax2.plot(t, g["beat_z_q"][k], color="#1f77b4", lw=0.8, alpha=0.6)
    ax2.set_ylabel("z_q (int8, = z x 16)", color="#1f77b4")
    ax.axvline(0, color="#999", lw=0.6)
    ax.set_title("input window: 360 samples (180 before / 180 after R)")
    for i in range(4):
        a = g[f"beat_conv{i}"][k]
        axh = fig.add_subplot(gs[1 + i // 2, 2 * (i % 2):2 * (i % 2) + 2])
        L = a.shape[1]
        im = axh.imshow(a, aspect="auto", cmap="viridis", vmin=0, vmax=max(int(a.max()), 1), interpolation="nearest",
                        extent=(-500, 500, a.shape[0] - 0.5, -0.5))
        c = qm.convs[i]
        axh.set_title(f"block {i + 1}: conv{i} {a.shape[0]} ch x {L} pos, int8 x 2^-{c['a_out']} "
                      f"(max {int(a.max())}/127)", fontsize=9)
        axh.set_ylabel("channel")
        axh.set_xlabel("ms from R")
        fig.colorbar(im, ax=axh, fraction=0.04, pad=0.02)
    axg = fig.add_subplot(gs[1:, 5])
    gap = g["beat_gap"][k]
    axg.imshow(gap[:, None], aspect="auto", cmap="magma", interpolation="nearest")
    axg.set_xticks([])
    axg.set_ylabel("GAP channel (0..63)")
    axg.set_title(f"GAP\n(sum of 45,\nmax {int(gap.max())})", fontsize=9)
    axl = fig.add_subplot(gs[0, 6])
    lg = g["beat_logits"][k]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]
    axl.bar(CLASSES, lg, color=colors)
    axl.axhline(0, color="#999", lw=0.6)
    axl.set_title(f"logits (int32, x 2^-{qm.fc2['k'] + qm.fc2['a_in']})\nargmax = {CLASSES[pred]}", fontsize=9)
    axr = fig.add_subplot(gs[1:, 6])
    axr.axis("off")
    rp = g["beat_rr_parts"][k]
    axr.text(0, 1, f"RR (samples, 360 Hz)\nrr_prev {rp[0]}\nrr_next {rp[1]}\nrr_avg {rp[2]}/{rp[3]}"
                   f" = {rp[2] / rp[3]:.1f}\nratio {rp[0] * rp[3] / rp[2]:.3f}\n\nrr_q (int8)\n"
                   f"{' '.join(str(int(v)) for v in g['beat_rr_q'][k])}\n\nFC1 (32, int8)\n"
                   f"nonzero {int(np.count_nonzero(g['beat_fc1'][k]))}/32\nmax {int(g['beat_fc1'][k].max())}",
             va="top", family="monospace", fontsize=9)
    fig.suptitle(f"{name}, beat {k}: true {LABEL_NAME[int(lab)]}, predicted {CLASSES[pred]} "
                 f"(frozen Lite, fully integer)", fontsize=13)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=90, bbox_inches="tight")
    plt.close(fig)


def first_of(cls: str) -> tuple[str, int]:
    c = CLASSES.index(cls)
    for seg in golden_manifest()["segments"]:
        g = load_golden_segment(seg["name"], arrays=("beat_label",))
        hit = np.flatnonzero(g["beat_label"] == c)
        if len(hit):
            return seg["name"], int(hit[0])
    raise SystemExit(f"no golden beat with label {cls}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", help="golden segment name (export/golden_manifest.json)")
    ap.add_argument("--beat", type=int, help="beat index inside the segment")
    ap.add_argument("--first-of", choices=CLASSES, help="first golden beat with this label")
    ap.add_argument("--out", default=str(REPO_ROOT / "viz"))
    args = ap.parse_args()
    if args.first_of:
        name, k = first_of(args.first_of)
    elif args.golden is not None and args.beat is not None:
        name, k = args.golden, args.beat
    else:
        raise SystemExit("give --golden and --beat, or --first-of")
    fz = load_frozen()
    g = load_golden_segment(name)
    if not 0 <= k < len(g["beat_pred"]):
        raise SystemExit(f"{name} has {len(g['beat_pred'])} beats")
    ok = recompute_matches(fz.qm, g, k)
    print_trace(name, k, fz.qm, g, stage_table(fz.qm, g, k), ok)
    path = Path(args.out) / f"trace_{name}_beat{k}.png"
    draw(name, k, fz.qm, g, path)
    print(f"saved {path}")
    if not ok:
        raise SystemExit("recomputed integer trace differs from the golden vectors")


if __name__ == "__main__":
    main()
