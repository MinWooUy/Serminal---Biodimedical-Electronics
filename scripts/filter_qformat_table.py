"""Table for choosing the Q-format of the fixed-point 0.5-40 Hz filter (no decision is taken here).

For each (F fractional coefficient bits, G guard bits): pole radii (stability), max frequency-response
deviation of the quantised coefficients vs float (dB, 0.5-40 Hz), time-domain error vs the float sosfilt
output on DS1 records (mV and relative to signal RMS), and the largest |accumulator| / |output| seen
(log2) -> whether int32 suffices (< 2^31) or int64 is required.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy import signal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from python.data.mitbih import load_record  # noqa: E402
from python.dsp.filters import apply_offline, design_bandpass  # noqa: E402
from python.quant.filter_int import pole_radii, quantize_sos, sos_filter_int  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default="101,108,207,208,232")
    ap.add_argument("--seconds", type=float, default=300.0, help="per record (start of the record)")
    ap.add_argument("--configs", default="12:0,14:0,14:2,14:4,16:2,20:4,24:4,28:4")
    args = ap.parse_args()
    sos = design_bandpass()
    f = np.linspace(0.5, 40.0, 400)
    _, h_ref = signal.sosfreqz(sos, worN=f, fs=360)
    recs = []
    for r in [int(v) for v in args.records.split(",")]:
        try:
            rec = load_record(r)
        except Exception as e:                       # DS2 ids are locked on purpose
            print(f"skip {r}: {type(e).__name__}")
            continue
        n = int(args.seconds * 360)
        x_int = rec.adc[:n].astype(np.int64) - rec.adc_zero
        recs.append((r, x_int, apply_offline(x_int / rec.adc_gain, sos), rec.adc_gain))
    print(f"records {[r for r, *_ in recs]}, {args.seconds:.0f} s each")
    print(f"{'F':>3} {'G':>3} | {'max pole r':>11} {'resp dev dB':>11} | {'max err mV':>10} {'rms err mV':>10} "
          f"{'err/RMS %':>9} | {'log2 max acc':>12} {'log2 max y':>10} {'int32?':>6}")
    for spec in args.configs.split(","):
        F, G = (int(v) for v in spec.split(":"))
        sq = quantize_sos(sos, F)
        r = pole_radii(sq, F)
        _, h_q = signal.sosfreqz(sq / 2.0**F, worN=f, fs=360)
        dev = np.max(np.abs(20 * np.log10(np.abs(h_q) / np.abs(h_ref))))
        errs, rms_sig, max_acc, max_y = [], [], 0, 0
        for _, x_int, y_ref, gain in recs:
            y, acc, ym = sos_filter_int(x_int, sq, F, G, track=True)
            errs.append(y / (2.0**G * gain) - y_ref)
            rms_sig.append(np.sqrt(np.mean(y_ref**2)))
            max_acc, max_y = max(max_acc, acc), max(max_y, ym)
        e = np.concatenate(errs)
        rel = 100 * np.sqrt(np.mean(e**2)) / np.mean(rms_sig)
        fits = max_acc < 2**31
        print(f"{F:>3} {G:>3} | {r.max():11.6f} {dev:11.4f} | {np.abs(e).max():10.5f} {np.sqrt(np.mean(e**2)):10.5f} "
              f"{rel:9.3f} | {np.log2(max(max_acc, 1)):12.2f} {np.log2(max(max_y, 1)):10.2f} {'yes' if fits else 'NO':>6}")


if __name__ == "__main__":
    main()
