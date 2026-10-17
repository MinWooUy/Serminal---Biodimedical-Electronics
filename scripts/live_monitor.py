"""Live monitor of the device output over UART (protocol: docs/uart_protocol.md). Observation only.

    python scripts/live_monitor.py --golden rec208_237600 [--speed 1|4] [--corrupt 0.02]   # simulated device
    python scripts/live_monitor.py --check all                                              # headless bit-exact check
    python scripts/live_monitor.py --serial COM5 [--baud 115200]                            # real board (needs pyserial)

GoldenSource simulates the firmware: the golden ADC segment (DS1) goes sample by sample through
IntegerPipelineStream (the streaming integer pipeline), every output is ENCODED into UART frames, and the
monitor only ever sees the DECODED byte stream — the same path a real board takes. STATS carries
beat_digest (CRC-32 over all BEAT payloads since the start); a final STATS follows the last sample.
--check runs this without a window. GATE 4 evidence: the device's final beat_digest equals the golden
digest on a clean line for a whole segment; the per-frame comparison with the golden .bin files is kept
for diagnosis (transport errors vs computation errors).
--corrupt drops one byte and inserts random garbage in that fraction of frames (resynchronisation demo).
Probabilities are computed on the host (softmax of logits / 2^logit_frac); the device sends int32 logits.
"""

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from python.viz import uart  # noqa: E402
from python.viz.common import CLASSES, load_frozen  # noqa: E402
from python.viz.golden import golden_manifest, load_golden_segment  # noqa: E402
from python.quant.pipeline_int import FILTER_GUARD_BITS  # noqa: E402
from python.viz.stream import IntegerPipelineStream  # noqa: E402

FS = 360
COLORS = ("#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd")


class GoldenSource:
    """Simulated device: golden ADC -> IntegerPipelineStream -> UART frames (bytes)."""

    def __init__(self, segment: str, fz=None, frame_samples: int = 8, stats_every: int = FS, corrupt: float = 0.0,
                 seed: int = 0, adc_gain: int = 200):
        self.fz = fz or load_frozen()
        g = load_golden_segment(segment, arrays=("adc",))
        self.adc = g["adc"].tolist()
        self.meta = g["meta"]
        self.stream = IntegerPipelineStream(self.fz.qm, self.fz.pt, self.fz.sos, self.meta["adc_zero"], self.fz.delay,
                                            self.fz.half, self.fz.rr_lo, self.fz.rr_hi)
        self.frame_samples, self.stats_every = frame_samples, stats_every
        self.logit_frac = int(self.fz.qm.fc2["k"] + self.fz.qm.fc2["a_in"])
        self.adc_gain = adc_gain
        self.corrupt, self.rng = corrupt, np.random.default_rng(seed)
        self.pos = 0
        self.n_beats = 0
        self.digest = 0
        self._pend_adc, self._pend_filt, self._pend_first = [], [], 0
        self._t0 = None
        self.done = False

    def _emit(self, fr: bytes, out: list):
        if self.corrupt and self.rng.random() < self.corrupt:
            b = bytearray(fr)
            del b[int(self.rng.integers(0, len(b)))]
            b += self.rng.integers(0, 256, size=int(self.rng.integers(1, 12))).astype(np.uint8).tobytes()
            fr = bytes(b)
        out.append(fr)

    def _flush_samples(self, out: list):
        if self._pend_adc:
            self._emit(uart.encode_samples(self._pend_first, self._pend_adc, self._pend_filt), out)
            self._pend_adc, self._pend_filt = [], []

    def _handle(self, events, out: list):
        for ev in events:
            if ev[0] == "sample":
                _, idx, adc, y = ev
                if not self._pend_adc:
                    self._pend_first = idx
                self._pend_adc.append(adc)
                self._pend_filt.append(y)
                if len(self._pend_adc) == self.frame_samples:
                    self._flush_samples(out)
                if (idx + 1) % self.stats_every == 0:
                    self._stats(idx, out)
            else:
                b = ev[1]
                self._flush_samples(out)                    # samples up to the beat's emission go first
                self.n_beats += 1
                flags = (uart.FLAG_SEARCHBACK if b.searchback else 0) | (uart.FLAG_REFINE_EDGE if b.edge else 0)
                beat = uart.Beat(b.r, b.emit_idx, b.pred, flags, b.logits, *b.rr_parts)
                self.digest = uart.crc32_update(self.digest, uart.beat_payload(beat))
                self._emit(uart.encode_beat(beat), out)

    def _stats(self, idx: int, out: list):
        self._flush_samples(out)
        self._emit(uart.encode_stats(uart.Stats(idx, self.n_beats, 0, 0, 0, self.adc_gain, FILTER_GUARD_BITS,
                                                self.logit_frac, self.digest)), out)

    def step(self, n_samples: int) -> bytes:
        out = []
        end = min(self.pos + n_samples, len(self.adc))
        for i in range(self.pos, end):
            self._handle(self.stream.push(self.adc[i]), out)
        self.pos = end
        if self.pos == len(self.adc) and not self.done:
            self._handle(self.stream.flush(), out)
            self._stats(len(self.adc) - 1, out)               # final STATS: digest of the whole run
            self.done = True
        return b"".join(out)

    def read(self, speed: float) -> bytes:
        """Paced read for the live view: as many samples as wall time x speed allows."""
        if self._t0 is None:
            self._t0 = time.perf_counter()
        target = int((time.perf_counter() - self._t0) * FS * speed)
        return self.step(max(0, min(target - self.pos, 4 * FS)))


class SerialSource:
    def __init__(self, port: str, baud: int):
        try:
            import serial                                   # pyserial: not in requirements (ask before adding)
        except ImportError as e:
            raise SystemExit("pyserial is not installed; it is needed only for --serial") from e
        self.ser = serial.Serial(port, baud, timeout=0)
        self.done = False

    def read(self, speed: float) -> bytes:
        return self.ser.read(4096)


class HostState:
    """What the monitor knows: decoded samples (ring or full), beats, last STATS. Decoding uses HostDecoder
    (CRC-16 + host plausibility checks); rejected frames are counted, never shown."""

    def __init__(self, keep: int | None = 8 * FS):
        self.dec = uart.HostDecoder()
        self.idx = [] if keep is None else deque(maxlen=keep)
        self.filt = [] if keep is None else deque(maxlen=keep)
        self.adc = [] if keep is None else deque(maxlen=keep)
        self.beats = [] if keep is None else deque(maxlen=200)
        self.packets = [] if keep is None else None          # every accepted packet (check mode)
        self.counts = [0, 0, 0, 0]
        self.stats = None
        self.ended = False

    @property
    def missing(self) -> int:
        return self.dec.samples_lost

    def feed(self, data: bytes):
        self._consume(self.dec.feed(data))

    def end(self):
        if not self.ended:
            self.ended = True
            self._consume(self.dec.end())

    def _consume(self, packets):
        if self.packets is not None:
            self.packets.extend(packets)
        for p in packets:
            if isinstance(p, uart.Samples):
                self.idx.extend(range(p.first_idx, p.first_idx + len(p.adc)))
                self.adc.extend(p.adc)
                self.filt.extend(p.filt)
            elif isinstance(p, uart.Beat):
                self.beats.append(p)
                self.counts[p.pred] += 1                    # pred in 0..3 is guaranteed by HostDecoder
            else:
                self.stats = p

    def probs(self, b: uart.Beat) -> np.ndarray | None:
        if self.stats is None:
            return None
        z = np.array(b.logits, dtype=np.float64) / 2.0 ** self.stats.logit_frac
        e = np.exp(z - z.max())
        return e / e.sum()


# ---------------------------------------------------------------- headless check against the golden vectors

def check_segment(name: str, fz, corrupt: float = 0.0, chunk_seed: int = 0, corrupt_seed: int = 0) -> dict:
    """Simulated device -> UART bytes (optionally corrupted) -> HostDecoder -> compare with the golden reference.

    GATE 4 evidence (clean line): the device's FINAL beat_digest (last STATS) == golden digest, same beat count.
    Diagnosis, per frame:
      transport errors  : frames rejected by the decoder, samples never received, golden beats never received;
      computation errors: ACCEPTED frames that differ from golden (SAMPLES: ADC / filter output; BEAT: the whole
                          payload incl. emit_idx and flags) or do not exist in golden. On a corrupted line such
                          frames are false accepts of damaged frames (robustness metric, not a gate)."""
    from python.viz.golden import golden_beats
    src = GoldenSource(name, fz, corrupt=corrupt, seed=corrupt_seed)
    host = HostState(keep=None)
    rng = np.random.default_rng(chunk_seed)
    while not src.done:
        data = src.step(int(rng.integers(1, 200)))
        k = 0
        while k < len(data):                                # arbitrary chunking, as a UART driver would
            m = int(rng.integers(1, 64))
            host.feed(data[k:k + m])
            k += m
    host.end()
    g = load_golden_segment(name, arrays=("adc", "filt"))
    n = len(g["adc"])
    gold_beats = golden_beats(name, fz)
    gold_digest = uart.beat_digest(gold_beats)
    gold_set = set(gold_beats)
    bad_frames = {"samples": 0, "beats": 0}
    seen_idx, seen_beats = set(), set()
    final = None
    for p in host.packets:
        if isinstance(p, uart.Samples):
            ii = np.arange(p.first_idx, p.first_idx + len(p.adc))
            ok = (ii >= 0).all() and (ii < n).all() and not (set(ii.tolist()) & seen_idx)
            ok = ok and np.array_equal(p.adc, g["adc"][ii]) and np.array_equal(p.filt, g["filt"][ii])
            bad_frames["samples"] += not ok
            seen_idx.update(ii.tolist())
        elif isinstance(p, uart.Beat):
            bad_frames["beats"] += p not in gold_set or p in seen_beats
            seen_beats.add(p)
        elif p.sample_idx == n - 1:
            final = p                                       # the final STATS of the run
    host_digest = uart.beat_digest([p for p in host.packets if isinstance(p, uart.Beat)])
    d = host.dec
    return {
        "digest": {"golden": gold_digest, "golden_beats": len(gold_beats),
                   "board": None if final is None else final.beat_digest,
                   "board_beats": None if final is None else final.n_beats,
                   "board_ok": final is not None and final.beat_digest == gold_digest
                   and final.n_beats == len(gold_beats),
                   "host_recomputed": host_digest, "host_ok": host_digest == gold_digest},
        "transport": {"frames_rejected": d.bad_header + d.bad_crc + d.transport_error, "bad_header": d.bad_header,
                      "bad_crc": d.bad_crc, "transport_error": d.transport_error, "skipped_bytes": d.skipped_bytes,
                      "missing_samples": n - len(seen_idx & set(range(n))),
                      "beats_not_received": len(gold_set - seen_beats)},
        "computation": bad_frames,
        "frames_ok": d.frames_ok, "samples": n, "beats_golden": len(gold_beats), "beats_received": len(seen_beats),
        "max_latency_ms": 1000 * src.stream.max_latency / FS,
    }


def check_ok(r: dict) -> bool:
    """GATE 4 on a clean line: board digest == golden and no computation error in the per-frame diagnosis."""
    return r["digest"]["board_ok"] and r["computation"]["samples"] == 0 and r["computation"]["beats"] == 0


# ---------------------------------------------------------------- live view

def run_live(source, speed: float, window_s: float = 8.0):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    host = HostState(keep=int(window_s * FS))
    fig = plt.figure(figsize=(14, 6))
    ax = fig.add_axes((0.05, 0.12, 0.66, 0.8))
    axp = fig.add_axes((0.76, 0.55, 0.21, 0.37))
    axt = fig.add_axes((0.76, 0.08, 0.21, 0.4))
    axt.axis("off")
    (line,) = ax.plot([], [], lw=0.9, color="#333")
    marks = ax.scatter([], [], s=40, zorder=3)
    labels = []
    bars = axp.bar(CLASSES, [0, 0, 0, 0], color=COLORS)
    axp.set_ylim(0, 1)
    axp.set_title("last beat: P(class)", fontsize=10)
    txt = axt.text(0, 1, "", va="top", family="monospace", fontsize=9)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("mV (0.5-40 Hz, device filter)")

    def update(_):
        host.feed(source.read(speed))
        if source.done:
            host.end()
        if not host.idx:
            return
        gain = host.stats.adc_gain if host.stats else 200
        frac = host.stats.filt_frac if host.stats else 12
        t = np.asarray(host.idx) / FS
        y = np.asarray(host.filt) / 2.0**frac / gain
        line.set_data(t, y)
        t1 = t[-1]
        ax.set_xlim(max(0.0, t1 - window_s), max(window_s, t1))
        lo, hi = (np.min(y), np.max(y)) if len(y) else (-1, 1)
        ax.set_ylim(lo - 0.2 * (hi - lo + 1e-3), hi + 0.35 * (hi - lo + 1e-3))
        vis = [b for b in host.beats if b.r_idx / FS >= t1 - window_s and b.r_idx >= host.idx[0]]
        for lb in labels:
            lb.remove()
        labels.clear()
        if vis:
            ia = np.asarray(host.idx)                       # may have gaps (lost frames)
            pos = np.clip(np.searchsorted(ia, [b.r_idx for b in vis]), 0, len(ia) - 1)
            yy = [y[k] if ia[k] == b.r_idx else np.nan for k, b in zip(pos, vis)]
            marks.set_offsets(np.c_[[b.r_idx / FS for b in vis], yy])
            marks.set_color([COLORS[b.pred] for b in vis])
            for b, v in zip(vis, yy):
                if np.isnan(v):
                    continue
                labels.append(ax.text(b.r_idx / FS, v + 0.12 * (hi - lo + 1e-3), CLASSES[b.pred], ha="center",
                                      color=COLORS[b.pred], fontsize=11, weight="bold"))
        else:
            marks.set_offsets(np.zeros((0, 2)))
        last = host.beats[-1] if host.beats else None
        p = host.probs(last) if last else None
        for bar, v in zip(bars, p if p is not None else [0, 0, 0, 0]):
            bar.set_height(v)
        s = host.stats
        cyc = ("n/a (PC simulation)" if s is None or s.cpu_hz == 0 else
               f"{s.cycles_last} ({1e3 * s.cycles_last / s.cpu_hz:.2f} ms), max {s.cycles_max}")
        d = host.dec
        lat = f"{1e3 * (last.emit_idx - last.r_idx) / FS:.0f} ms" if last else "-"
        txt.set_text(f"beats   N {host.counts[0]}  S {host.counts[1]}\n        V {host.counts[2]}  F {host.counts[3]}\n"
                     f"latency {lat}\ncycles  {cyc}\nframes  ok {d.frames_ok}\n"
                     f"errors  header {d.bad_header}  CRC {d.bad_crc}\n"
                     f"        transport_error {d.transport_error}\n"
                     f"digest  {'-' if s is None else f'0x{s.beat_digest:08X} ({s.n_beats} beats)'}\n"
                     f"skipped {d.skipped_bytes} B, samples_lost {d.samples_lost}, resyncs {d.resyncs}")
        ax.set_title(f"t = {t1:.1f} s, speed x{speed:g}" + ("  (end of source)" if source.done else ""))

    anim = FuncAnimation(fig, update, interval=50, cache_frame_data=False)
    plt.show()
    return anim


def main() -> None:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--golden", help="golden segment name, e.g. rec208_237600 (see export/golden_manifest.json)")
    src.add_argument("--serial", help="serial port of the board (needs pyserial)")
    src.add_argument("--check", help="headless bit-exact check: segment name or 'all'")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--speed", type=float, choices=(1, 4), default=1)
    ap.add_argument("--corrupt", type=float, default=0.0, help="fraction of frames corrupted (golden source)")
    args = ap.parse_args()
    if args.check:
        fz = load_frozen()
        names = [sg["name"] for sg in golden_manifest()["segments"]] if args.check == "all" else [args.check]
        clean_line = args.corrupt == 0
        ok_all, tot_tr, tot_cp, n_board_ok, n_host_ok = True, {}, {}, 0, 0
        for name in names:
            r = check_segment(name, fz, corrupt=args.corrupt)
            ok_all &= check_ok(r)
            dg = r["digest"]
            n_board_ok += dg["board_ok"]
            n_host_ok += dg["host_ok"]
            for k, v in r["transport"].items():
                tot_tr[k] = tot_tr.get(k, 0) + v
            for k, v in r["computation"].items():
                tot_cp[k] = tot_cp.get(k, 0) + v
            exact = check_ok(r) and all(v == 0 for v in r["transport"].values()) and dg["host_ok"]
            board = "not received" if dg["board"] is None else f"0x{dg['board']:08X} ({dg['board_beats']} beats)"
            print(f"{name}: {'BIT-EXACT' if exact else ('DIGEST OK' if dg['board_ok'] else 'DIGEST MISMATCH')}"
                  f"  frames {r['frames_ok']}, beats {r['beats_received']}/{r['beats_golden']}, "
                  f"max beat latency {r['max_latency_ms']:.0f} ms\n"
                  f"    beat_digest: golden 0x{dg['golden']:08X} ({dg['golden_beats']} beats), board {board}, "
                  f"host recomputed 0x{dg['host_recomputed']:08X} -> board {'==' if dg['board_ok'] else '!='} golden, "
                  f"host {'==' if dg['host_ok'] else '!='} golden\n"
                  f"    {'computation errors' if clean_line else 'false accepts     '}: SAMPLES frames "
                  f"{r['computation']['samples']}, BEAT frames {r['computation']['beats']}\n"
                  f"    transport errors  : " + ", ".join(f"{k} {v}" for k, v in r["transport"].items()))
        print(f"TOTAL ({len(names)} segments, corrupt = {args.corrupt}): board digest == golden {n_board_ok}/{len(names)}, "
              f"host-recomputed digest == golden {n_host_ok}/{len(names)}; "
              f"{'computation errors' if clean_line else 'false accepts'} {tot_cp}; transport errors {tot_tr}")
        if clean_line:
            print(f"GATE 4 (clean line: board beat_digest == golden on every segment, 0 computation errors): "
                  f"{'PASS' if ok_all else 'FAIL'}")
            raise SystemExit(0 if ok_all else 1)
        print("robustness run (--corrupt): reported only, not a gate")
        return
    source = SerialSource(args.serial, args.baud) if args.serial else GoldenSource(args.golden, corrupt=args.corrupt)
    run_live(source, args.speed)


if __name__ == "__main__":
    main()
