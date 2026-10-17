"""Classification metrics (4 classes N/S/V/F).

- macro-F1 over N/S/V/F (main metric) and per-class P/R/F1 + confusion matrix
- macro-F1 over N/S/V only (sklearn labels=[N, S, V] on all beats; true F beats
  still count as false positives when predicted N/S/V)
- AAMI EC57-style class statistics from the confusion matrix C (rows = true, cols = pred):
    SVEB Se = C[S,S] / sum C[S,:]       SVEB +P = C[S,S] / sum C[:,S]
    VEB  Se = C[V,V] / sum C[V,:]       VEB  +P = C[V,V] / (sum C[:,V] - C[F,V])
  i.e. for VEB, fusion beats classified as V are not counted as false positives
  (EC57 convention, as in de Chazal et al. 2004). No Q class here.
"""

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

CLASSES = ("N", "S", "V", "F")
LABELS = list(range(len(CLASSES)))
N, S, V, F = LABELS


def ec57(cm: np.ndarray) -> dict:
    cm = np.asarray(cm, dtype=float)

    def ratio(a, b):
        return float(a / b) if b > 0 else 0.0

    return {"SVEB": {"se": ratio(cm[S, S], cm[S].sum()), "ppv": ratio(cm[S, S], cm[:, S].sum())},
            "VEB": {"se": ratio(cm[V, V], cm[V].sum()), "ppv": ratio(cm[V, V], cm[:, V].sum() - cm[F, V])}}


def report(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=LABELS, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    return {
        "macro_f1": float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)),
        "macro_f1_nsv": float(f1_score(y_true, y_pred, labels=[N, S, V], average="macro", zero_division=0)),
        "ec57": ec57(cm),
        "per_class": {c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]),
                          "support": int(s[i])} for i, c in enumerate(CLASSES)},
        "confusion": cm.tolist(),
        "n_pred_N_as_F": int(cm[N, F]),
    }


def format_report(rep: dict) -> str:
    e = rep["ec57"]
    lines = [f"macro-F1 {rep['macro_f1']:.4f}   macro-F1 (N/S/V) {rep['macro_f1_nsv']:.4f}   "
             f"SVEB Se {e['SVEB']['se']:.4f} +P {e['SVEB']['ppv']:.4f}   "
             f"VEB Se {e['VEB']['se']:.4f} +P {e['VEB']['ppv']:.4f}",
             f"{'cls':>3} {'P':>7} {'R':>7} {'F1':>7} {'n':>6}"]
    for c, v in rep["per_class"].items():
        lines.append(f"{c:>3} {v['precision']:7.4f} {v['recall']:7.4f} {v['f1']:7.4f} {v['support']:>6}")
    lines.append("confusion (rows = true N/S/V/F, cols = pred):")
    lines += ["  " + " ".join(f"{v:>6}" for v in row) for row in rep["confusion"]]
    return "\n".join(lines)


def mean_std(reports: list[dict]) -> dict:
    """mean ± std over seeds of macro-F1 (4 and 3 classes), EC57 SVEB/VEB Se/+P and per-class P/R/F1."""
    out = {"macro_f1": _ms([r["macro_f1"] for r in reports]),
           "macro_f1_nsv": _ms([r["macro_f1_nsv"] for r in reports]),
           "ec57": {g: {m: _ms([r["ec57"][g][m] for r in reports]) for m in ("se", "ppv")}
                    for g in ("SVEB", "VEB")},
           "n_pred_N_as_F": _ms([r["n_pred_N_as_F"] for r in reports])}
    for c in CLASSES:
        out[c] = {m: _ms([r["per_class"][c][m] for r in reports]) for m in ("precision", "recall", "f1")}
    return out


MIN_MEAN_DELTA = 0.01


def paired_by_seed(base: list[float], cand: list[float], min_mean: float = MIN_MEAN_DELTA) -> dict:
    """Paired comparison over the same seeds (pooled-OOF macro-F1). Adopt the candidate only if
    BOTH hold: delta > 0 for every seed AND mean delta >= min_mean (0.01)."""
    d = np.asarray(cand, dtype=float) - np.asarray(base, dtype=float)
    all_pos, big_enough = bool(np.all(d > 0)), bool(d.mean() >= min_mean)
    return {"delta": d.tolist(), "mean": float(d.mean()), "std": float(d.std(ddof=1)) if len(d) > 1 else 0.0,
            "all_seeds_positive": all_pos, "mean_ge_min": big_enough, "adopt": all_pos and big_enough}


def _ms(v) -> dict:
    v = np.asarray(v, dtype=float)
    return {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0, "n": int(len(v))}
