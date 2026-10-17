import numpy as np
import pytest

from python.data.svdb import ANN_SCALE, ChannelEval, choose_channel, scale_ann, to_360


def test_resample_ratio_and_annotation_scaling():
    assert ANN_SCALE == 2.8125
    x = np.sin(2 * np.pi * 1.0 * np.arange(1280) / 128)       # 10 s at 128 Hz
    y = to_360(x)
    assert len(y) == 3600
    # a 1 Hz sine stays a 1 Hz sine: zero crossings every 180 samples at 360 Hz
    np.testing.assert_allclose(y[900:2700], np.sin(2 * np.pi * np.arange(900, 2700) / 360), atol=1e-3)
    np.testing.assert_array_equal(scale_ann(np.array([0, 1, 128, 1000])), [0, 3, 360, 2812])


def test_drop_out_of_range_annotation():
    from python.data.svdb import drop_out_of_range
    s, sym = drop_out_of_range(np.array([10, 230399, 230400]), ["N", "V", "N"], 230400)
    assert s.tolist() == [10, 230399] and sym == ["N", "V"]


def _e(ch, se, ppv, pos):
    return ChannelEval(ch, se, ppv, 100, 0, 0, pos, 20.0)


def test_choose_channel_rule_option_a():
    # positive channel preferred even if the other one has a higher Se
    best, why = choose_channel([_e(0, 0.995, 0.99, 0.3), _e(1, 0.990, 0.99, 0.8)])
    assert best.ch == 1 and why == "ok"
    # both positive -> higher Se, tie -> higher +P
    assert choose_channel([_e(0, 0.99, 0.98, 0.9), _e(1, 0.99, 0.99, 0.7)])[0].ch == 1
    # no mostly-positive channel -> record excluded (802-like)
    assert choose_channel([_e(0, 0.99, 0.99, 0.2), _e(1, 0.99, 0.99, 0.4)])[0] is None
    # positive channel below 97% -> excluded even if the negative one is perfect (801/805-like)
    best, why = choose_channel([_e(0, 0.997, 0.892, 0.87), _e(1, 0.999, 1.0, 0.15)])
    assert best is None and "below" in why


def test_negative_fallback_rule_a5b():
    from python.data.svdb import choose_negative_fallback
    # 801-like: positive channel below 97% is NOT a fallback candidate; the negative one is used
    best, _ = choose_negative_fallback([_e(0, 0.997, 0.892, 0.87), _e(1, 0.999, 1.0, 0.15)])
    assert best.ch == 1
    # no negative channel >= 97% -> still excluded
    assert choose_negative_fallback([_e(0, 0.96, 0.99, 0.1), _e(1, 0.99, 0.95, 0.2)])[0] is None
    # both negative and passing -> higher Se
    assert choose_negative_fallback([_e(0, 0.995, 0.99, 0.1), _e(1, 0.999, 0.98, 0.2)])[0].ch == 1


@pytest.mark.needs_data
def test_a5b_keeps_a5_sample_unchanged():
    from python.data.beats import load_beats  # noqa: F401
    from python.data.mitbih import REPO_ROOT
    from python.data.svdb import load_svdb_train, svdb_cache_dir, svdb_neg_cache_dir
    from python.experiment import load_config
    bcfg = load_config(REPO_ROOT / "configs/beats.yaml")
    det = load_config(REPO_ROOT / bcfg["detector_config"])
    main, neg = svdb_cache_dir(bcfg, det), svdb_neg_cache_dir(bcfg, det)
    if not (main / "manifest.json").exists() or not (neg / "manifest.json").exists():
        pytest.skip("SVDB caches not built")
    a5 = load_svdb_train(main, "n25")
    a5b = load_svdb_train(main, "n25", extra_caches=(neg,))
    n = len(a5["y"])
    for k in a5:
        assert np.array_equal(a5b[k][:n], a5[k]), k                  # A5b = A5 sample + additions
    assert len(a5b["y"]) > n


@pytest.mark.needs_data
def test_svdb_channel_record_shape():
    from python.data.svdb import available_records, load_channel
    recs = available_records()
    if not recs:
        pytest.skip("no SVDB records downloaded")
    rec = load_channel(recs[0], 0)
    assert rec.fs == 360 and rec.adc_zero == 0 and rec.adc_gain == 200.0
    assert np.array_equal(rec.signal, rec.adc.astype(np.int64) / 200.0)
    assert rec.r_samples.max() < len(rec.signal) and np.all(np.diff(rec.r_samples) > 0)
