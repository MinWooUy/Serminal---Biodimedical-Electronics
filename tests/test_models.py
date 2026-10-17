import numpy as np
import pytest

torch = pytest.importorskip("torch")

from python.models.cnn1d import FULL, LITE, ECGCNN, n_params  # noqa: E402
from python.models.metrics import report  # noqa: E402
from python.models.training import (  # noqa: E402
    SPLIT_BUFFER, SPLIT_TRAIN, SPLIT_VAL, TrainConfig, assert_ds1, class_weights, fit, fold_indices,
    inner_split, predict_proba, rr_stats, set_seed,
)

DELAY, HALF, FS = 2, 180, 360


# ---------- model ----------

@pytest.mark.parametrize("cfg,head_in,hidden", [(FULL, 516, 256), (LITE, 68, 32)])
def test_shapes_and_head(cfg, head_in, hidden):
    m = ECGCNN(cfg).eval()
    assert m.fc1.in_features == head_in and m.fc1.out_features == hidden and m.fc2.out_features == 4
    out = m(torch.zeros(3, 1, 360), torch.zeros(3, 4))
    assert out.shape == (3, 4)


@pytest.mark.parametrize("cfg,head_in", [(FULL, 518), (LITE, 70)])
def test_six_rr_features_head(cfg, head_in):
    import dataclasses
    m = ECGCNN(dataclasses.replace(cfg, n_rr=6)).eval()
    assert m.fc1.in_features == head_in and m.rr_mean.shape == (6,)
    assert m(torch.zeros(2, 1, 360), torch.zeros(2, 6)).shape == (2, 4)


def test_param_counts():
    # conv (no bias) + BN (2 per channel) + fc1 + fc2
    def expected(ch, ks, hidden):
        n, c_in = 0, 1
        for c, k in zip(ch, ks):
            n += c_in * c * k + 2 * c
            c_in = c
        return n + (c_in + 4) * hidden + hidden + hidden * 4 + 4
    assert n_params(ECGCNN(FULL)) == expected(FULL.channels, FULL.kernels, 256)
    assert n_params(ECGCNN(LITE)) == expected(LITE.channels, LITE.kernels, 32)


def test_dropout_placement():
    m = ECGCNN(LITE)
    names = [type(x).__name__ for x in m.features]
    # blocks 1-3: Conv BN ReLU MaxPool Dropout; block 4: Conv BN ReLU
    assert names == ["Conv1d", "BatchNorm1d", "ReLU", "MaxPool1d", "Dropout"] * 3 + ["Conv1d", "BatchNorm1d", "ReLU"]
    assert all(x.p == 0.2 for x in m.features if isinstance(x, torch.nn.Dropout))
    assert m.drop.p == 0.5


def test_rr_stats_in_checkpoint_and_applied():
    m = ECGCNN(LITE).eval()
    m.set_rr_stats([1, 2, 3, 4], [2, 2, 2, 2])
    sd = m.state_dict()
    assert torch.equal(sd["rr_mean"], torch.tensor([1., 2., 3., 4.]))
    x = torch.randn(2, 1, 360)
    rr = torch.randn(2, 4)
    m2 = ECGCNN(LITE).eval()
    m2.load_state_dict(sd)
    assert torch.allclose(m(x, rr), m2(x, rr))


# ---------- statistics ----------

def test_class_weights_sqrt_inverse():
    y = np.array([0] * 400 + [1] * 100 + [2] * 25 + [3] * 4)
    w = class_weights(y)
    raw = 1 / np.sqrt([400, 100, 25, 4])
    np.testing.assert_allclose(w, raw * 4 / raw.sum())
    assert abs(w.mean() - 1) < 1e-12
    with pytest.raises(ValueError):
        class_weights(np.array([0, 1, 2]))


def test_class_weight_variants():
    y = np.array([0] * 400 + [1] * 100 + [2] * 25 + [3] * 4)
    np.testing.assert_array_equal(class_weights(y, "none"), np.ones(4))
    inv = 1 / np.array([400, 100, 25, 4])
    np.testing.assert_allclose(class_weights(y, "inverse"), inv * 4 / inv.sum())
    cap = class_weights(y, "sqrt_inverse_capF")
    assert cap[3] <= cap[1] + 1e-12 and abs(cap.mean() - 1) < 1e-12
    with pytest.raises(ValueError):
        class_weights(y, "log")


def test_weighted_loss():
    from python.models.training import WeightedLoss
    torch.manual_seed(0)
    logits, y = torch.randn(64, 4), torch.randint(0, 4, (64,))
    w = np.array([0.2, 1.3, 0.7, 1.8])
    ref = torch.nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32))(logits, y)
    assert torch.equal(WeightedLoss(w, "ce")(logits, y), ref)
    assert torch.allclose(WeightedLoss(w, "focal", gamma=0.0)(logits, y), ref, atol=1e-6)
    assert WeightedLoss(w, "focal", gamma=2.0)(logits, y) < ref       # focal down-weights easy beats
    with pytest.raises(ValueError):
        WeightedLoss(w, "hinge")


def test_rr_stats():
    rr = np.random.default_rng(0).normal(size=(100, 4))
    mu, sd = rr_stats(rr)
    np.testing.assert_allclose(mu, rr.mean(0))
    np.testing.assert_allclose(sd, rr.std(0))


def test_ec57_and_nsv():
    from python.models.metrics import ec57, paired_by_seed
    # rows true N/S/V/F, cols pred
    cm = np.array([[90, 5, 3, 2],
                   [4, 16, 0, 0],
                   [1, 0, 30, 1],
                   [2, 0, 6, 2]])
    e = ec57(cm)
    assert e["SVEB"]["se"] == 16 / 20 and e["SVEB"]["ppv"] == 16 / 21
    assert e["VEB"]["se"] == 30 / 32 and e["VEB"]["ppv"] == 30 / (39 - 6)     # F->V not a VEB FP
    y_true = np.repeat(np.arange(4), cm.sum(1))
    y_pred = np.concatenate([np.repeat(np.arange(4), row) for row in cm])
    rep = report(y_true, y_pred)
    assert rep["confusion"] == cm.tolist() and rep["n_pred_N_as_F"] == 2
    from sklearn.metrics import f1_score
    assert abs(rep["macro_f1_nsv"] - f1_score(y_true, y_pred, labels=[0, 1, 2], average="macro")) < 1e-12
    p = paired_by_seed([0.40, 0.42, 0.41], [0.43, 0.425, 0.40])
    assert not p["adopt"] and abs(p["mean"] - (0.03 + 0.005 - 0.01) / 3) < 1e-12
    assert paired_by_seed([0.4, 0.4, 0.4], [0.41, 0.42, 0.401])["adopt"]          # mean +0.0103
    small = paired_by_seed([0.4, 0.4, 0.4], [0.405, 0.408, 0.402])                 # all > 0, mean +0.005
    assert small["all_seeds_positive"] and not small["mean_ge_min"] and not small["adopt"]


def test_metrics_match_sklearn():
    from sklearn.metrics import f1_score
    rng = np.random.default_rng(1)
    yt, yp = rng.integers(0, 4, 500), rng.integers(0, 4, 500)
    rep = report(yt, yp)
    assert abs(rep["macro_f1"] - f1_score(yt, yp, average="macro")) < 1e-12
    assert np.array(rep["confusion"]).sum() == 500


# ---------- splits ----------

def _fake_dataset(rng, recs=(101, 106, 108, 109, 112, 114, 115, 116, 118, 119), rec_len=20000):
    rec_id, r_pos = [], []
    for r in recs:
        pos = np.cumsum(rng.integers(200, 400, size=200))
        pos = pos[(pos > 200) & (pos < rec_len - 200)]
        rec_id += [r] * len(pos)
        r_pos += pos.tolist()
    return np.array(rec_id), np.array(r_pos), {r: rec_len for r in recs}


def test_inner_split_no_window_straddles_cut():
    rec_len, buf = 20000, 360
    r = np.arange(200, 19800, 37)
    code = inner_split(r, rec_len, DELAY, HALF, 0.2, buf)
    cut = 16000
    start, end = r + DELAY - HALF, r + DELAY + HALF
    assert np.all(end[code == SPLIT_TRAIN] <= cut - buf // 2)
    assert np.all(start[code == SPLIT_VAL] >= cut + buf // 2)
    # everything whose window touches [cut - 180, cut + 180) is buffer
    touches = (end > cut - buf // 2) & (start < cut + buf // 2)
    assert np.all(code[touches] == SPLIT_BUFFER)
    assert set(np.unique(code)) == {SPLIT_TRAIN, SPLIT_VAL, SPLIT_BUFFER}


def test_fold_indices_isolation():
    rng = np.random.default_rng(2)
    rec_id, r_pos, rec_len = _fake_dataset(rng)
    folds = [[101, 106], [108, 109], [112, 114], [115, 116], [118, 119]]
    cfg = TrainConfig()
    for k in range(5):
        idx = fold_indices(rec_id, r_pos, rec_len, folds, k, DELAY, HALF, cfg, FS)
        outer, tr, iv, bf = (set(idx[n].tolist()) for n in ("outer", "train", "inner_val", "buffer"))
        # partition of all beats
        assert outer | tr | iv | bf == set(range(len(rec_id)))
        assert not (outer & tr) and not (outer & iv) and not (tr & iv) and not (bf & (tr | iv | outer))
        # no outer-fold record in train / inner-val
        assert not set(rec_id[list(tr | iv)].tolist()) & set(folds[k])
        assert set(rec_id[list(outer)].tolist()) == set(folds[k])
        # inner-val beats are all after the cut of their record
        cut = int(round(20000 * 0.8))
        assert np.all(r_pos[idx["inner_val"]] + DELAY - HALF >= cut + 180)
        assert np.all(r_pos[idx["train"]] + DELAY + HALF <= cut - 180)


@pytest.mark.needs_data
def test_fold_indices_real_cache():
    from python.models.cv_data import load_cv_data
    try:
        cv = load_cv_data("configs/beats.yaml")
    except FileNotFoundError:
        pytest.skip("beat cache not built")
    d, cfg = cv.data, TrainConfig()
    for k in range(5):
        idx = fold_indices(d["rec_id"], d["r_pos"], cv.rec_len, cv.folds, k, cv.delay, cv.half, cfg, cv.fs)
        used = np.concatenate([idx["train"], idx["inner_val"]])
        assert not set(d["rec_id"][used].tolist()) & set(cv.folds[k])         # no outer-fold beat
        assert not set(used.tolist()) & set(idx["buffer"].tolist())           # no buffer beat
        assert not set(idx["train"].tolist()) & set(idx["inner_val"].tolist())
        for r in set(d["rec_id"][used].tolist()):
            cut = int(round(cv.rec_len[r] * 0.8))
            tr = idx["train"][d["rec_id"][idx["train"]] == r]
            iv = idx["inner_val"][d["rec_id"][idx["inner_val"]] == r]
            assert np.all(d["r_pos"][tr] + cv.delay + cv.half <= cut - 180)
            assert np.all(d["r_pos"][iv] + cv.delay - cv.half >= cut + 180)
        assert len(idx["outer"]) + len(used) + len(idx["buffer"]) == len(d["y"])


def test_extra_train_beats_only_in_inner_train():
    from python.models.cv_data import add_extra_train, with_extra_train
    rng = np.random.default_rng(6)
    rec_id, r_pos, rec_len = _fake_dataset(rng)
    n = len(rec_id)
    main = {"y": rng.integers(0, 4, n).astype(np.int8), "rr": rng.normal(size=(n, 4)), "rec_id": rec_id,
            "r_pos": r_pos}
    m = 500
    extra = {"y": np.full(m, 1, dtype=np.int8), "rr": rng.normal(5.0, 1.0, size=(m, 4)),
             "rec_id": np.full(m, 800), "r_pos": np.arange(m) * 300, "X_only_in_extra": np.zeros(m)}
    data, n_main, extra_idx = add_extra_train(main, extra)
    assert n_main == n and "X_only_in_extra" not in data and len(data["y"]) == n + m
    folds = [[101, 106], [108, 109], [112, 114], [115, 116], [118, 119]]
    for k in range(5):
        idx = fold_indices(data["rec_id"][:n_main], data["r_pos"][:n_main], rec_len, folds, k, DELAY, HALF,
                           TrainConfig(), FS)
        idx = with_extra_train(idx, extra_idx, n_main)
        for part in ("outer", "inner_val", "buffer"):
            assert np.all(idx[part] < n_main)                        # no extra beat outside inner-train
        assert set(extra_idx.tolist()) <= set(idx["train"].tolist())
        # class weights and RR mu/sigma are computed on the combined inner-train
        ytr = data["y"][idx["train"]]
        assert np.sum(ytr == 1) >= m
        mu, _ = rr_stats(data["rr"][idx["train"]])
        assert mu[0] > rr_stats(data["rr"][idx["train"][idx["train"] < n_main]])[0][0]
    with pytest.raises(ValueError):
        with_extra_train({"outer": np.array([n_main]), "inner_val": np.array([], dtype=int),
                          "buffer": np.array([], dtype=int), "train": np.array([0])}, extra_idx, n_main)


def test_all_records_split_for_final_training():
    # train_final.py: one 'fold' with an empty outer set -> every record is time-split
    rng = np.random.default_rng(4)
    rec_id, r_pos, rec_len = _fake_dataset(rng)
    idx = fold_indices(rec_id, r_pos, rec_len, [[]], 0, DELAY, HALF, TrainConfig(), FS)
    assert len(idx["outer"]) == 0
    assert set(rec_id[idx["train"]].tolist()) == set(rec_id.tolist())
    assert set(rec_id[idx["inner_val"]].tolist()) == set(rec_id.tolist())
    assert len(idx["train"]) + len(idx["inner_val"]) + len(idx["buffer"]) == len(rec_id)


def test_assert_ds1():
    assert_ds1(np.array([101, 230]))
    with pytest.raises(ValueError):
        assert_ds1(np.array([101, 100]))


# ---------- deterministic training (CPU only) ----------

def _toy_data(n=400, seed=0):
    rng = np.random.default_rng(seed)
    y = np.tile(np.arange(4), n // 4)
    X = rng.normal(size=(n, 360)).astype(np.float32)
    X[:, 170:190] += y[:, None].astype(np.float32)            # learnable signal
    rr = rng.normal(size=(n, 4)).astype(np.float32) + y[:, None]
    return {"X": X, "rr": rr, "y": y.astype(np.int8)}


def _train_once(seed):
    set_seed(seed, deterministic=True)
    data = _toy_data()
    m = ECGCNN(LITE)
    m.set_rr_stats(*rr_stats(data["rr"][:300]))
    cfg = TrainConfig(max_epochs=2, batch_size=64)
    hist = fit(m, data, np.arange(300), np.arange(300, 400), cfg, np.ones(4), seed, "cpu", log=lambda s: None)
    return m, hist, predict_proba(m, data, np.arange(300, 400), "cpu")


def test_training_deterministic_cpu():
    m1, h1, p1 = _train_once(7)
    m2, h2, p2 = _train_once(7)
    for k, v in m1.state_dict().items():
        assert torch.equal(v, m2.state_dict()[k]), k
    assert [h["val_loss"] for h in h1] == [h["val_loss"] for h in h2]
    assert np.array_equal(p1, p2)
    torch.use_deterministic_algorithms(False)


def test_diag_does_not_change_training():
    def run(with_diag):
        set_seed(11, deterministic=True)
        data = _toy_data()
        m = ECGCNN(LITE)
        m.set_rr_stats(*rr_stats(data["rr"][:300]))
        diag = (lambda mm, e: {"p": predict_proba(mm, data, np.arange(300, 400), "cpu").mean()}) if with_diag else None
        h = fit(m, data, np.arange(300), np.arange(300, 360), TrainConfig(max_epochs=3, batch_size=64),
                np.ones(4), 11, "cpu", log=lambda s: None, diag=diag)
        return m, h
    m1, h1 = run(False)
    m2, h2 = run(True)
    for k, v in m1.state_dict().items():
        assert torch.equal(v, m2.state_dict()[k]), k
    assert [h["val_loss"] for h in h1] == [h["val_loss"] for h in h2]
    assert all("diag" in h for h in h2) and not any("diag" in h for h in h1)
    torch.use_deterministic_algorithms(False)


def test_fixed_epochs_keeps_last_weights():
    set_seed(5, deterministic=True)
    data = _toy_data()
    m = ECGCNN(LITE)
    m.set_rr_stats(*rr_stats(data["rr"][:300]))
    seen = {}
    cfg = TrainConfig(max_epochs=4, batch_size=64, early_stopping=False, early_stop_patience=1)

    def diag(mm, e):
        seen[e] = {k: v.detach().clone() for k, v in mm.state_dict().items()}
        return {}
    h = fit(m, data, np.arange(300), np.arange(300, 360), cfg, np.ones(4), 5, "cpu", log=lambda s: None, diag=diag)
    assert len(h) == 4 and sorted(seen) == [1, 2, 3, 4]
    for k, v in m.state_dict().items():
        assert torch.equal(v, seen[4][k]), k                 # last epoch, not the best one
    assert TrainConfig.from_dict({"snapshot_epochs": [3, 5]}).snapshot_epochs == (3, 5)
    torch.use_deterministic_algorithms(False)


def _wide_and_x(n=8, half=180, pad=5, seed=0):
    from python.data.beats import extract_wide, extract_windows, normalize_beats
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.normal(size=5000))
    r = np.array([400, 1000, 1700, 2500, 3100, 3800, 4200, 4500])[:n]
    W = extract_wide(x, r, 2, half, pad)
    win, keep = extract_windows(x, r, 2, half)
    assert keep.all()
    return W, normalize_beats(win, 1e-6, 8.0), x, r


def test_wide_window_centre_is_deployment_window():
    from python.data.beats import extract_wide
    W, X, x, r = _wide_and_x()
    assert W.shape == (8, 370)
    np.testing.assert_array_equal(W[:, 5:365], x[(r + 2)[:, None] + np.arange(-180, 180)])
    edge = extract_wide(x, np.array([182]), 0, 180, 5)             # centre window starts at sample 2
    assert edge.shape == (1, 370) and np.all(edge[0, :4] == x[0]) and edge[0, 4] == x[1]
    with pytest.raises(ValueError):
        extract_wide(x, np.array([3]), 0, 180, 5)                    # incomplete centre window


def test_augment_batch_identity_and_jitter():
    from python.models.training import augment_batch
    W, X, _, _ = _wide_and_x()
    Wt = torch.as_tensor(W, dtype=torch.float32)
    off = TrainConfig(augment=True, aug_p=0.0)
    out = augment_batch(Wt, off, torch.Generator().manual_seed(0))
    assert out.shape == (8, 1, 360)
    np.testing.assert_allclose(out[:, 0].numpy(), X, atol=1e-4)      # p = 0 -> pipeline window
    on = TrainConfig(augment=True, aug_p=1.0, aug_noise_sigma=0.0, aug_wander_amp=0.0)
    a = augment_batch(Wt, on, torch.Generator().manual_seed(1))
    b = augment_batch(Wt, on, torch.Generator().manual_seed(1))
    assert torch.equal(a, b)                                          # deterministic per generator
    shifts = []
    for i in range(8):
        cands = [int(np.argmin([np.abs(a[i, 0].numpy() - _z(W[i, 5 + s:365 + s])).max()
                                for s in range(-5, 6)])) - 5]
        shifts += cands
    assert all(-5 <= s <= 5 for s in shifts)
    full = TrainConfig(augment=True, aug_p=1.0, aug_noise_sigma=10.0)
    assert augment_batch(Wt, full, torch.Generator().manual_seed(2)).abs().max() <= 8.0   # clipped


def _z(w):
    return np.clip((w - w.mean()) / (w.std() + 1e-6), -8, 8)


def test_augmentation_only_in_training():
    # predictions never depend on augmentation settings (they use the cached X)
    data = _toy_data()
    data["X_wide"] = np.pad(data["X"], ((0, 0), (5, 5)), mode="edge")
    set_seed(3, deterministic=True)
    m = ECGCNN(LITE)
    m.set_rr_stats(*rr_stats(data["rr"][:300]))
    cfg = TrainConfig(max_epochs=1, batch_size=64, augment=True)
    fit(m, data, np.arange(300), np.arange(300, 360), cfg, np.ones(4), 3, "cpu", log=lambda s: None)
    p1 = predict_proba(m, data, np.arange(300, 400), "cpu")
    data2 = dict(data, X_wide=data["X_wide"] * 0)                    # X_wide is not read at prediction
    p2 = predict_proba(m, data2, np.arange(300, 400), "cpu")
    assert np.array_equal(p1, p2)
    torch.use_deterministic_algorithms(False)


def test_predict_proba_rows_sum_to_one():
    _, _, p = _train_once(3)
    np.testing.assert_allclose(p.sum(1), 1, atol=1e-5)
    torch.use_deterministic_algorithms(False)
