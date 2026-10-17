import numpy as np
from scipy import signal

from python.dsp.filters import (
    SosStream, apply_offline, design_bandpass, group_delay_compensation, group_delay_samples,
)


def test_stream_matches_sosfilt_bitwise():
    rng = np.random.default_rng(0)
    x = rng.normal(size=5000)
    sos = design_bandpass()
    st = SosStream(sos)
    on = np.array([st.push(v) for v in x])
    assert np.array_equal(on, apply_offline(x, sos))


def test_causal():
    rng = np.random.default_rng(1)
    x = rng.normal(size=2000)
    y = rng.normal(size=2000)
    y[:1000] = x[:1000]           # same past, different future
    sos = design_bandpass()
    np.testing.assert_array_equal(apply_offline(x, sos)[:1000], apply_offline(y, sos)[:1000])


def test_passband_edges():
    sos = design_bandpass()
    f, h = signal.sosfreqz(sos, worN=[0.5, 10.0, 40.0], fs=360)
    mag_db = 20 * np.log10(np.abs(h))
    np.testing.assert_allclose(mag_db[[0, 2]], -3.01, atol=0.1)
    assert abs(mag_db[1]) < 0.1


def test_group_delay_10hz():
    sos = design_bandpass()
    gd = group_delay_samples(sos, 10.0)
    print(f"group delay @10 Hz: {gd:.3f} samples = {gd / 360 * 1000:.2f} ms")
    assert 2.0 < gd < 3.0
    assert group_delay_compensation(sos) == 2
