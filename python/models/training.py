"""Record-grouped CV training with a time-based inner validation split.

For outer fold k: the outer-val records (fold k) are used ONLY to produce
out-of-fold predictions. Each remaining (train) record is split in time:
the first 80% -> inner-train, the last 20% -> inner-val, with a 1 s buffer
centred on the cut. A beat's window is [R + delay - half, R + delay + half);
it is inner-train only if the window ends at or before cut - buffer/2, inner-val
only if it starts at or after cut + buffer/2, otherwise it is dropped (buffer),
so no window straddles the boundary.

Limitation: inner-val beats come from the same patients as inner-train beats,
so early stopping is tuned on seen patients (optimistic vs. new patients).
RR features of the first inner-val beats still use R times from before the cut
(timing only, no signal samples).
"""

import copy
import random
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from python.data.constants import DS1
from python.models.cnn1d import N_CLASSES

SPLIT_TRAIN, SPLIT_VAL, SPLIT_BUFFER = 0, 1, 2


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 256
    max_epochs: int = 60
    lr: float = 1e-3
    weight_decay: float = 1e-2
    class_weight: str = "sqrt_inverse"   # none | inverse | sqrt_inverse | sqrt_inverse_capF
    loss: str = "ce"                     # ce | focal
    focal_gamma: float = 2.0
    plateau_factor: float = 0.5
    plateau_patience: int = 3
    early_stop_patience: int = 6
    inner_val_frac: float = 0.2
    inner_buffer_s: float = 1.0
    early_stopping: bool = True          # False: train exactly max_epochs, keep the LAST weights
    snapshot_epochs: tuple[int, ...] = ()  # epochs at which train_cv stores outer-fold predictions
    # augmentation (TRAINING ONLY; evaluation/OOF always use the cached centre window X)
    augment: bool = False
    aug_p: float = 0.5                   # independent probability of each transform per beat
    aug_jitter: int = 5                  # R jitter, samples (crop from X_wide)
    aug_noise_sigma: float = 0.05        # Gaussian noise, z units
    aug_wander_amp: float = 0.3          # baseline wander max amplitude, z units
    aug_wander_hz: tuple[float, float] = (0.1, 0.5)
    norm_eps: float = 1e-6               # must equal beats.beat_norm (re-normalisation after the crop)
    norm_clip: float = 8.0

    @classmethod
    def from_dict(cls, d: dict) -> "TrainConfig":
        d = dict(d)
        if "snapshot_epochs" in d:
            d["snapshot_epochs"] = tuple(int(e) for e in d["snapshot_epochs"])
        if "aug_wander_hz" in d:
            d["aug_wander_hz"] = tuple(float(v) for v in d["aug_wander_hz"])
        return cls(**d)


def augment_batch(W: torch.Tensor, cfg: TrainConfig, gen: torch.Generator, fs: int = 360) -> torch.Tensor:
    """Training-time augmentation from un-normalised wide windows W (B, 2*half + 2*pad).

    1. R jitter: with prob p crop at offset pad + s, s ~ U{-J..J}; otherwise the centre crop (s = 0)
    2. per-beat z-score with the pipeline's eps (same as the cached X), then
    3. baseline wander (prob p): A sin(2 pi f t + phi), A ~ U(0, amp), f ~ U(f_lo, f_hi) Hz
    4. Gaussian noise (prob p): sigma in z units
    5. clip to +-norm_clip (the deployment input is always clipped)
    Returns (B, 1, 2*half).
    """
    B, L = W.shape
    pad = cfg.aug_jitter
    n = L - 2 * pad
    dev = W.device

    def u(*shape):
        return torch.rand(*shape, generator=gen, device=dev)

    s = torch.randint(-pad, pad + 1, (B,), generator=gen, device=dev) * (u(B) < cfg.aug_p)
    idx = (pad + s)[:, None] + torch.arange(n, device=dev)[None, :]
    x = torch.gather(W, 1, idx)
    x = (x - x.mean(1, keepdim=True)) / (x.std(1, unbiased=False, keepdim=True) + cfg.norm_eps)
    t = torch.arange(n, device=dev, dtype=x.dtype)[None, :] / fs
    f = cfg.aug_wander_hz[0] + (cfg.aug_wander_hz[1] - cfg.aug_wander_hz[0]) * u(B, 1)
    wander = cfg.aug_wander_amp * u(B, 1) * torch.sin(2 * torch.pi * f * t + 2 * torch.pi * u(B, 1))
    x = x + wander * (u(B, 1) < cfg.aug_p)
    noise = cfg.aug_noise_sigma * torch.randn(B, n, generator=gen, device=dev)
    x = x + noise * (u(B, 1) < cfg.aug_p)
    return x.clamp(-cfg.norm_clip, cfg.norm_clip)[:, None, :]


# ---------------------------------------------------------------- splits

def inner_split(r_pos: np.ndarray, rec_len: int, delay: int, half: int, frac: float,
                buffer: int) -> np.ndarray:
    """Per-beat code for one record: 0 inner-train, 1 inner-val, 2 buffer (dropped)."""
    cut = int(round(rec_len * (1.0 - frac)))
    start = np.asarray(r_pos, dtype=np.int64) + delay - half
    end = start + 2 * half                                    # exclusive
    code = np.full(len(start), SPLIT_BUFFER, dtype=np.int8)
    code[end <= cut - buffer // 2] = SPLIT_TRAIN
    code[start >= cut + buffer // 2] = SPLIT_VAL
    return code


def fold_indices(rec_id: np.ndarray, r_pos: np.ndarray, rec_len: dict[int, int], folds: list[list[int]],
                 k: int, delay: int, half: int, cfg: TrainConfig, fs: int = 360) -> dict[str, np.ndarray]:
    """Indices (into the beat arrays) of outer-val, inner-train, inner-val and buffer beats."""
    outer_recs = set(folds[k])
    is_outer = np.isin(rec_id, list(outer_recs))
    code = np.full(len(rec_id), -1, dtype=np.int8)
    buffer = int(round(cfg.inner_buffer_s * fs))
    for r in np.unique(rec_id[~is_outer]):
        m = rec_id == r
        code[m] = inner_split(r_pos[m], rec_len[int(r)], delay, half, cfg.inner_val_frac, buffer)
    return {"outer": np.flatnonzero(is_outer),
            "train": np.flatnonzero(code == SPLIT_TRAIN),
            "inner_val": np.flatnonzero(code == SPLIT_VAL),
            "buffer": np.flatnonzero(code == SPLIT_BUFFER)}


# ---------------------------------------------------------------- statistics

CLASS_WEIGHTS = ("none", "inverse", "sqrt_inverse", "sqrt_inverse_capF")
LOSSES = ("ce", "focal")
S_IDX, F_IDX = 1, 3


def class_weights(y: np.ndarray, kind: str = "sqrt_inverse") -> np.ndarray:
    """Class weights from inner-train counts, normalised to mean 1.
    none: all 1; inverse: 1/n_c; sqrt_inverse: 1/sqrt(n_c);
    sqrt_inverse_capF: sqrt_inverse with w_F = min(w_F, w_S) (then renormalised)."""
    counts = np.bincount(y, minlength=N_CLASSES).astype(float)
    if np.any(counts == 0):
        raise ValueError(f"a class is missing from the training set: {counts}")
    if kind == "none":
        return np.ones(N_CLASSES)
    if kind == "inverse":
        w = 1.0 / counts
    elif kind in ("sqrt_inverse", "sqrt_inverse_capF"):
        w = 1.0 / np.sqrt(counts)
        if kind == "sqrt_inverse_capF":
            w[F_IDX] = min(w[F_IDX], w[S_IDX])
    else:
        raise ValueError(f"unknown class_weight {kind!r}")
    return w * (N_CLASSES / w.sum())                          # mean weight = 1


class WeightedLoss(nn.Module):
    """Class-weighted CE or focal loss, reduced as sum_i w_{y_i} l_i / sum_i w_{y_i}
    (same normalisation as nn.CrossEntropyLoss(weight=...)); focal with gamma = 0 equals CE."""

    def __init__(self, weights: np.ndarray, kind: str = "ce", gamma: float = 2.0):
        super().__init__()
        if kind not in LOSSES:
            raise ValueError(f"unknown loss {kind!r}")
        self.kind, self.gamma = kind, float(gamma)
        self.register_buffer("weight", torch.as_tensor(weights, dtype=torch.float32))

    def forward(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.kind == "ce":        # exactly nn.CrossEntropyLoss(weight=...) as used by A0/A1
            return nn.functional.cross_entropy(logits, y, weight=self.weight)
        logp = torch.log_softmax(logits, dim=1).gather(1, y[:, None]).squeeze(1)
        loss = -logp
        if self.kind == "focal":
            loss = loss * (1.0 - logp.exp()).clamp_min(0.0) ** self.gamma
        w = self.weight[y]
        return (w * loss).sum() / w.sum()


def rr_stats(rr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return rr.mean(axis=0), np.maximum(rr.std(axis=0), 1e-6)


# ---------------------------------------------------------------- training

def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if deterministic:
        torch.use_deterministic_algorithms(True)


def _batches(n: int, bs: int, gen: torch.Generator | None):
    order = torch.randperm(n, generator=gen) if gen is not None else torch.arange(n)
    for i in range(0, n, bs):
        yield order[i:i + bs]


@torch.no_grad()
def _eval_loss(model, X, R, Y, loss_fn, bs) -> float:
    model.eval()
    tot, n = 0.0, 0
    w = loss_fn.weight
    for idx in _batches(len(Y), bs, None):
        logits = model(X[idx], R[idx])
        tot += float(loss_fn(logits, Y[idx])) * float(w[Y[idx]].sum())
        n += float(w[Y[idx]].sum())
    return tot / n                                            # weighted mean over the whole set


def fit(model: nn.Module, data: dict, idx_train: np.ndarray, idx_val: np.ndarray, cfg: TrainConfig,
        weights: np.ndarray, seed: int, device: str, log=print, diag=None) -> list[dict]:
    """Train with weighted CE; LR plateau + early stopping on weighted inner-val loss.
    Restores the best (lowest inner-val loss) weights. Returns the per-epoch history.

    With cfg.early_stopping = False the model trains for exactly max_epochs (the LR plateau
    scheduler on inner-val loss still runs) and the last weights are kept.

    diag: optional callable(model, epoch) -> dict, evaluated once per epoch AFTER the
    scheduler / early-stopping bookkeeping and stored in the history under
    "diag". DIAGNOSTIC ONLY (e.g. outer-fold macro-F1): its result is never read
    by the scheduler, early stopping or checkpoint selection. It runs in eval
    mode under no_grad and does not touch the shuffling generator.

    With cfg.augment = True each training batch is re-built from X_wide by augment_batch;
    the inner-val loss, diag and all predictions use the cached centre window X.
    """
    def tensors(idx):
        return (torch.as_tensor(data["X"][idx][:, None, :], dtype=torch.float32, device=device),
                torch.as_tensor(data["rr"][idx], dtype=torch.float32, device=device),
                torch.as_tensor(data["y"][idx].astype(np.int64), device=device))

    Xt, Rt, Yt = tensors(idx_train)
    Xv, Rv, Yv = tensors(idx_val)
    model.to(device)
    loss_fn = WeightedLoss(weights, cfg.loss, cfg.focal_gamma).to(device)   # also the inner-val loss
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=cfg.plateau_factor,
                                                       patience=cfg.plateau_patience)
    gen = torch.Generator().manual_seed(seed)
    if cfg.augment:
        # separate generator: augmentation must not change the shuffling order
        aug_gen = torch.Generator(device=device).manual_seed(seed + 100_003)
        Wt = torch.as_tensor(data["X_wide"][idx_train], dtype=torch.float32, device=device)
        if Wt.shape[1] - 2 * cfg.aug_jitter != Xt.shape[2]:
            raise ValueError("X_wide padding does not match aug_jitter")
    best, best_state, wait, history = float("inf"), None, 0, []
    for epoch in range(1, cfg.max_epochs + 1):
        t0 = time.time()
        model.train()
        tot, n = 0.0, 0
        for idx in _batches(len(Yt), cfg.batch_size, gen):
            idx = idx.to(device)
            opt.zero_grad(set_to_none=True)
            xb = augment_batch(Wt[idx], cfg, aug_gen) if cfg.augment else Xt[idx]
            loss = loss_fn(model(xb, Rt[idx]), Yt[idx])
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(idx)
            n += len(idx)
        val = _eval_loss(model, Xv, Rv, Yv, loss_fn, 1024)
        sched.step(val)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        rec = {"epoch": epoch, "train_loss": tot / n, "val_loss": val,
               "lr": opt.param_groups[0]["lr"], "time_s": time.time() - t0}
        history.append(rec)
        improved = val < best
        if improved:
            best, wait = val, 0
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        else:
            wait += 1
        msg = (f"    epoch {epoch:3d}  train {rec['train_loss']:.4f}  inner-val {val:.4f}"
               f"{' *' if improved else '  '}  lr {rec['lr']:.1e}  {rec['time_s']:.2f}s")
        if diag is not None:                       # diagnostic only, see docstring
            rec["diag"] = diag(model, epoch)
            msg += f"  | diag {rec['diag'].get('msg', '')}"
        log(msg)
        if cfg.early_stopping and wait >= cfg.early_stop_patience:
            break
    if cfg.early_stopping:
        model.load_state_dict(best_state)
    return history


@torch.no_grad()
def predict_proba(model: nn.Module, data: dict, idx: np.ndarray, device: str, bs: int = 1024) -> np.ndarray:
    model.eval().to(device)
    out = []
    for i in range(0, len(idx), bs):
        j = idx[i:i + bs]
        x = torch.as_tensor(data["X"][j][:, None, :], dtype=torch.float32, device=device)
        r = torch.as_tensor(data["rr"][j], dtype=torch.float32, device=device)
        out.append(torch.softmax(model(x, r), dim=1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, N_CLASSES), dtype=np.float32)


def assert_ds1(rec_id: np.ndarray) -> None:
    extra = set(np.unique(rec_id).tolist()) - set(DS1)
    if extra:
        raise ValueError(f"training data must be DS1 only; found {sorted(extra)}")
