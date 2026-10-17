import numpy as np
import pytest

torch = pytest.importorskip("torch")

from python.models.cnn1d import FULL, LITE, ECGCNN  # noqa: E402
from python.quant.int8 import (  # noqa: E402
    conv_int, float_forward, fold_bn, fold_model, int_forward, pow2_exp, quantize_input, quantize_model,
    rshift_round,
)


# ---------- golden requantization ----------

@pytest.mark.parametrize("acc,s,expected", [
    (5, 1, 3),       # 2.5  -> 3 (half rounds up)
    (-5, 1, -2),     # -2.5 -> -2 ((-5 + 1) >> 1 = -4 >> 1 = -2)
    (-6, 2, -1),     # -1.5 -> -1 ((-6 + 2) >> 2 = -1)
    (-7, 2, -2),     # -1.75 -> -2 ((-7 + 2) >> 2 = -5 >> 2 = floor(-1.25) = -2)
    (-1, 3, 0),      # -0.125 -> 0 ((-1 + 4) >> 3 = 0)
    (-12, 3, -1),    # -1.5 -> -1 ((-12 + 4) >> 3 = -8 >> 3 = -1)
    (-13, 3, -2),    # -1.625 -> -2 ((-13 + 4) >> 3 = -9 >> 3 = floor(-1.125) = -2)
    (7, 0, 7),       # s = 0: unchanged
    (-3, -2, -12),   # s < 0: left shift
])
def test_rshift_round_hand_values(acc, s, expected):
    assert int(rshift_round(np.array([acc]), s)[0]) == expected


def test_rshift_round_matches_formula_on_negatives():
    acc = np.arange(-5000, 5000, dtype=np.int64)
    for s in range(1, 12):
        hand = np.array([(int(a) + (1 << (s - 1))) >> s for a in acc])     # Python >> = arithmetic shift
        assert np.array_equal(rshift_round(acc, s), hand)


def test_pow2_exp():
    assert pow2_exp(1.0) == 6            # 1 * 2^6 = 64 <= 127 < 128
    assert pow2_exp(0.5) == 7
    assert pow2_exp(127.0) == 0 and pow2_exp(200.0) == -1


# ---------- one small integer conv layer vs. hand computation ----------

def test_conv_int_matches_hand_loops():
    rng = np.random.default_rng(0)
    cin, cout, k, L = 2, 3, 3, 8
    x = rng.integers(-127, 128, size=(1, cin, L)).astype(np.int64)
    w = rng.integers(-127, 128, size=(cout, cin, k)).astype(np.int64)
    b = rng.integers(-3000, 3000, size=cout).astype(np.int64)
    kk = np.array([5, 6, 7])
    layer = {"w_q": w, "b_q": b, "k": kk, "a_in": 4, "a_out": 2, "pool": False}
    got = conv_int(x, layer)
    for o in range(cout):
        s = int(kk[o]) + 4 - 2
        for t in range(L):
            acc = int(b[o])
            for c in range(cin):
                for j in range(k):
                    p = t + j - k // 2
                    if 0 <= p < L:
                        acc += int(w[o, c, j]) * int(x[0, c, p])
            y = (acc + (1 << (s - 1))) >> s
            assert got[0, o, t] == min(max(y, 0), 127), (o, t)


def test_quantize_input_rounding_and_saturation():
    x = np.array([[0.03, -0.03, 7.9, 8.0, -8.0, 0.031249]])
    assert quantize_input(x, 4).tolist() == [[0, 0, 126, 127, -127, 0]]


# ---------- folding ----------

def _random_bn_model(cfg, seed=0):
    torch.manual_seed(seed)
    m = ECGCNN(cfg)
    for mod in m.modules():
        if isinstance(mod, torch.nn.BatchNorm1d):
            mod.running_mean.uniform_(-0.5, 0.5)
            mod.running_var.uniform_(0.5, 2.0)
            mod.weight.data.uniform_(0.5, 1.5)
            mod.bias.data.uniform_(-0.2, 0.2)
    m.set_rr_stats([0.8, 0.8, 0.8, 1.0], [0.2, 0.2, 0.17, 0.15])
    return m.eval()


def test_fold_bn_equivalence():
    m = _random_bn_model(LITE)
    conv, bn = m.features[0], m.features[1]
    w, b = fold_bn(conv, bn)
    x = torch.randn(4, 1, 360)
    ref = bn(conv(x)).detach().double().numpy()
    from python.quant.int8 import _conv_same
    got = _conv_same(x.double().numpy(), w) + b[None, :, None]
    np.testing.assert_allclose(got, ref, atol=1e-5)


@pytest.mark.parametrize("cfg", [LITE, FULL])
def test_folded_float_matches_torch(cfg):
    m = _random_bn_model(cfg)
    x = torch.randn(16, 1, 360)
    rr = torch.rand(16, 4) + 0.5
    ref = m(x, rr).detach().double().numpy()
    got = float_forward(fold_model(m), x[:, 0].double().numpy(), rr.double().numpy())
    np.testing.assert_allclose(got, ref, atol=1e-4)            # GAP 1/45 and RR mu/sigma folded into FC1


def test_int_forward_close_to_float_and_no_overflow():
    m = _random_bn_model(LITE, seed=1)
    rng = np.random.default_rng(1)
    x = np.clip(rng.normal(size=(300, 360)), -8, 8)
    rr = rng.uniform(0.3, 1.5, size=(300, 4))
    fm = fold_model(m)
    qm = quantize_model(fm, x[:200], rr[:200], "max")
    li = int_forward(qm, x[200:], rr[200:])                    # raises on int32 overflow
    lf = float_forward(fm, x[200:], rr[200:])
    assert li.dtype == np.int64
    # dequantised integer logits approximate the float logits (an untrained random model has near-tied
    # logits, so argmax agreement is not a meaningful check here; the real check is on trained models)
    deq = li * 2.0 ** -(qm.fc2["k"] + qm.fc2["a_in"])
    err = np.abs(deq - lf).max() / np.abs(lf).max()
    assert err < 0.1, err
    assert np.corrcoef(deq.ravel(), lf.ravel())[0, 1] > 0.99
    for layer in qm.convs:
        assert np.abs(layer["w_q"]).max() <= 127
    # regression: GAP weights of FC1 (divided by 45) use their own exponent and the full int8 range
    assert np.abs(qm.fc1["wg_q"]).max(axis=1).min() >= 64


def test_save_load_quant_model_roundtrip(tmp_path):
    from python.quant.int8 import load_quant_model, save_quant_model
    m = _random_bn_model(LITE, seed=2)
    rng = np.random.default_rng(2)
    x = np.clip(rng.normal(size=(120, 360)), -8, 8)
    rr = rng.uniform(0.3, 1.5, size=(120, 4))
    qm = quantize_model(fold_model(m), x[:80], rr[:80])
    save_quant_model(qm, tmp_path / "q.npz")
    qm2 = load_quant_model(tmp_path / "q.npz")
    assert np.array_equal(int_forward(qm, x[80:], rr[80:]), int_forward(qm2, x[80:], rr[80:]))


def test_trace_matches_int_forward_q():
    from python.quant.int8 import int_forward_q, int_forward_trace, quantize_input, quantize_rr
    m = _random_bn_model(LITE, seed=3)
    rng = np.random.default_rng(3)
    x = np.clip(rng.normal(size=(60, 360)), -8, 8)
    rr = rng.uniform(0.3, 1.5, size=(60, 4))
    qm = quantize_model(fold_model(m), x[:40], rr[:40])
    xq, rq = quantize_input(x[40:], 4), quantize_rr(rr[40:], qm.rr_exp)
    tr = int_forward_trace(qm, xq, rq)
    assert np.array_equal(tr["logits"], int_forward_q(qm, xq, rq))
    assert tr["conv0"].shape == (20, 16, 360) and tr["pool2"].shape == (20, 64, 45) and tr["gap"].shape == (20, 64)
    assert tr["fc1"].min() >= 0 and tr["fc1"].max() <= 127
