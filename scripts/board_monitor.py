"""Live monitor with the CNN on the FPGA (layer_cmd_controller protocol): the PC does the pre-processing.

    python scripts/board_monitor.py --serial COM5 --golden rec208_237600            # real board, live window
    python scripts/board_monitor.py --sim --golden rec208_237600 [--sim-pace 1]      # no board: software FPGA model
    python scripts/board_monitor.py --sim --check rec208_237600                      # headless: digest vs golden
    python scripts/board_monitor.py --serial COM5 --check rec208_237600              # same on the real board

ECG source: the golden MIT-BIH ADC segment is replayed sample by sample at the real sampling rate (--speed 1 =
real time).  Pre-processing (filter, Pan-Tompkins, beat window, z-score, RR features) runs on the PC in
IntegerPipelineStream.  Every beat is then classified ON THE FPGA (host/board_classifier.py): the six layers are
sent over UART and only the 17 B result (4 logits + argmax) comes back.  The window shows the FPGA's answer.

Because a beat takes about 0.3 s on the wire (+ USB latency) the board is driven by a worker thread: the ECG
trace keeps running in real time and each label appears when the FPGA answers (the "board" line of the status
panel shows the queue and the round-trip time).  The software model (int_forward_q) classifies the same beat
in parallel and is only used to count disagreements with the FPGA.

--check replays the whole segment through the board as fast as the link allows and compares the final
beat_digest (CRC-32 over every beat's payload incl. the 4 FPGA logits) with the golden one: equal digests = the
FPGA returned exactly the golden logits for every beat.
"""

import argparse
import queue
import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from host.board_classifier import BoardClassifier, BoardError  # noqa: E402
from python.link.layer_host import LayerHost  # noqa: E402
from python.quant.pipeline_int import FILTER_GUARD_BITS  # noqa: E402
from python.viz import uart  # noqa: E402
from python.viz.common import load_frozen  # noqa: E402
from python.viz.golden import golden_manifest  # noqa: E402
from scripts import live_monitor as lm  # noqa: E402
from scripts.pipeline_int_eval import quantise_rr_parts  # noqa: E402

FS = lm.FS


class BoardSource(lm.GoldenSource):
    """Replayed ADC -> PC pre-processing -> FPGA classification -> UART-style frames for the monitor.

    sync=True : each beat is classified at once (headless --check; deterministic, no thread).
    sync=False: a worker thread drives the board; frames for a beat are emitted when its answer arrives.
    pace > 0  : (simulator only) the worker sleeps the simulated link time x pace so the demo behaves like the
                real link; with a real serial port the wall clock is the real thing and pace stays 0.
    max_queue : > 0 drops a new beat (counted as `skipped`) while that many beats are still waiting.
    retries   : how often a FAILED BEAT is sent again from scratch after recover().  Safe because a beat starts
                with WRITE_X + SWAP and rewrites everything; single commands are never retried (LayerHost)."""

    def __init__(self, segment: str, clf: BoardClassifier, fz=None, sync: bool = False, pace: float = 0.0,
                 max_queue: int = 0, retries: int = 1, **kw):
        super().__init__(segment, fz, **kw)
        self.clf, self.sync, self.pace, self.max_queue, self.retries = clf, sync, pace, max_queue, retries
        self.retried = 0
        self.hw = clf.hw
        self._eof = False
        self.pending = 0                                  # beats submitted and not yet emitted (main thread only)
        self.failed = self.skipped = self.mismatch = 0
        self.last_error = ""
        self.cycles_last = self.cycles_max = 0
        self.times = deque(maxlen=20)                     # last board round trips (s)
        self.t_last_result = None
        self._jobs: queue.Queue = queue.Queue()
        self._done: queue.Queue = queue.Queue()
        self._worker = None
        if not sync:
            self._worker = threading.Thread(target=self._work, daemon=True, name="board-worker")
            self._worker.start()

    # ---- board side (worker thread, or the caller in sync mode)
    def _classify(self, b, rr_q):
        t0 = time.perf_counter()
        for attempt in range(self.retries + 1):
            try:
                res = self.clf.classify(b.z_q, rr_q)
                break
            except BoardError as exc:
                self.clf.recover()
                if attempt == self.retries:
                    return exc
                self.retried += 1
        if self.pace > 0:
            time.sleep(max(0.0, res.seconds * self.pace - (time.perf_counter() - t0)))
        return res

    def _work(self):
        while True:
            job = self._jobs.get()
            if job is None:
                return
            b, rr_q = job
            self._done.put((b, self._classify(b, rr_q)))

    def close(self):
        if self._worker is not None:
            self._jobs.put(None)
            self._worker.join(timeout=5)

    # ---- PC side (main thread)
    def _handle(self, events, out: list):
        for ev in events:
            if ev[0] != "beat":
                super()._handle([ev], out)
                continue
            b = ev[1]
            self._flush_samples(out)                      # samples up to the beat's emission go first
            rr_q = quantise_rr_parts(np.array([b.rr_parts], dtype=np.int64), self.fz.qm.rr_exp, FS)[0]
            if self.sync:
                self._emit_result(b, self._classify(b, rr_q), out)
            elif self.max_queue and self.pending >= self.max_queue:
                self.skipped += 1
            else:
                self.pending += 1
                self._jobs.put((b, rr_q))

    def _emit_result(self, b, res, out: list):
        if isinstance(res, BoardError):
            self.failed += 1
            self.last_error = str(res)
            return
        self.times.append(res.seconds)
        self.cycles_last, self.cycles_max = res.cycles, max(self.cycles_max, res.cycles)
        self.mismatch += tuple(res.logits) != tuple(b.logits)
        self.t_last_result = time.perf_counter()
        self.n_beats += 1
        flags = (uart.FLAG_SEARCHBACK if b.searchback else 0) | (uart.FLAG_REFINE_EDGE if b.edge else 0)
        beat = uart.Beat(b.r, b.emit_idx, res.pred, flags, tuple(res.logits), *b.rr_parts)
        self.digest = uart.crc32_update(self.digest, uart.beat_payload(beat))
        self._emit(uart.encode_beat(beat), out)

    def _drain(self, out: list):
        while True:
            try:
                b, res = self._done.get_nowait()
            except queue.Empty:
                return
            self.pending -= 1
            self._emit_result(b, res, out)

    def _stats(self, idx: int, out: list):
        self._flush_samples(out)
        self._emit(uart.encode_stats(uart.Stats(idx, self.n_beats, self.cycles_last, self.cycles_max,
                                                self.hw.clk_hz, self.adc_gain, FILTER_GUARD_BITS, self.logit_frac,
                                                self.digest)), out)

    def step(self, n_samples: int) -> bytes:
        out = []
        end = min(self.pos + n_samples, len(self.adc))
        for i in range(self.pos, end):
            self._handle(self.stream.push(self.adc[i]), out)
        self.pos = end
        if self.pos == len(self.adc) and not self._eof:
            self._handle(self.stream.flush(), out)
            self._eof = True
        self._drain(out)
        if self._eof and self.pending == 0 and not self.done:
            self._stats(len(self.adc) - 1, out)            # final STATS: digest of the whole run
            self.done = True
            self.close()
        return b"".join(out)

    def status_text(self) -> str:
        avg = np.mean(self.times) if self.times else float("nan")
        ms = f"{1e3 * self.cycles_last / self.hw.clk_hz:.1f} ms" if self.cycles_last else "-"
        age = "-" if self.t_last_result is None else f"{time.perf_counter() - self.t_last_result:.1f} s ago"
        line = [f"board   {self.hw.rows}x{self.hw.cols} PE {self.hw.clk_hz / 1e6:g} MHz",
                f"queue   {self.pending} waiting, {avg:.2f} s/beat",
                f"FPGA    {ms} compute/beat",
                f"answer  {age}",
                f"vs SW   {self.mismatch} mismatch",
                f"link    failed {self.failed}, retried {self.retried}",
                f"        skipped {self.skipped}"]
        if self.last_error:
            line.append("error   " + self.last_error[-28:])
        return "\n".join(line)


# ---------------------------------------------------------------- transports

class _SimTransport:
    """Software FPGA behind the transport interface LayerHost uses (virtual clock, no sleeping)."""

    def __init__(self, baud: int):
        from python.link.layersim import HwConfig, LayerFpgaSim, LayerLink
        self.link = LayerLink(LayerFpgaSim(HwConfig(baud=baud)))

    def write(self, data: bytes) -> None:
        self.link.write(data)

    def read(self, timeout: float) -> bytes:
        return self.link.read(timeout)

    def wait(self, seconds: float) -> None:
        self.link.wait(seconds)

    def clock(self) -> float:
        return self.link.clock()

    def close(self) -> None:
        pass


def open_board(args):
    if args.sim:
        transport = _SimTransport(args.baud)
    else:
        from python.link.serial_link import SerialTransport
        transport = SerialTransport(args.serial, args.baud)
    host = LayerHost(transport, timeout=args.timeout)
    return transport, BoardClassifier(host)


# ---------------------------------------------------------------- headless check

def check_board(names, clf, fz, retries: int = 1) -> bool:
    ok_all = True
    for name in names:
        src = BoardSource(name, clf, fz, sync=True, retries=retries)
        t0 = time.perf_counter()
        r = lm.check_segment(name, fz, source=src)
        dg = r["digest"]
        ok = lm.check_ok(r) and src.failed == 0 and src.mismatch == 0
        ok_all &= ok
        avg = np.mean(src.times) if src.times else float("nan")
        verdict = "BOARD == GOLDEN (bit-exact)" if ok else "MISMATCH"
        board = "-" if dg["board"] is None else "0x%08X" % dg["board"]
        print(f"{name}: {verdict}  beats {r['beats_received']}/{r['beats_golden']}, "
              f"digest golden 0x{dg['golden']:08X} board {board}, "
              f"failed {src.failed}, board!=SW {src.mismatch}; link {avg:.2f} s/beat, {time.perf_counter() - t0:.0f} s wall")
        if src.last_error:
            print("   last error:", src.last_error)
    print(f"CHECK {'PASS' if ok_all else 'FAIL'}")
    return ok_all


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    link = ap.add_mutually_exclusive_group(required=True)
    link.add_argument("--serial", help="serial port of the board, e.g. COM5 (needs pyserial)")
    link.add_argument("--sim", action="store_true", help="software FPGA model instead of a board")
    what = ap.add_mutually_exclusive_group(required=True)
    what.add_argument("--golden", help="segment to replay, e.g. rec208_237600 (see export/golden_manifest.json)")
    what.add_argument("--check", help="headless bit-exact check: segment name or 'all'")
    ap.add_argument("--baud", type=int, default=1500000, help="must match the bitstream")
    ap.add_argument("--timeout", type=float, default=3.0, help="per-command response timeout (s)")
    ap.add_argument("--speed", type=float, default=1.0, help="replay speed, 1 = real time (the board needs ~0.3-1 s/beat)")
    ap.add_argument("--sim-pace", type=float, default=0.0, help="--sim: sleep the simulated link time x this factor per beat")
    ap.add_argument("--retries", type=int, default=1, help="resend a failed beat from scratch this many times")
    ap.add_argument("--max-queue", type=int, default=0, help="drop beats while this many wait for the board (0 = never)")
    args = ap.parse_args()

    fz = load_frozen()
    transport, clf = open_board(args)
    try:
        print(f"FPGA INFO: {clf.hw.rows}x{clf.hw.cols} PE, {clf.hw.clk_hz / 1e6:g} MHz, {clf.hw.baud} baud, "
              f"MAX_PAYLOAD {clf.hw.max_payload}, features 0x{clf.hw.features:02X}")
        if clf.hw.baud != args.baud:
            print(f"WARNING: INFO says {clf.hw.baud} baud, the port was opened at {args.baud}")
        if args.check:
            names = [s["name"] for s in golden_manifest()["segments"]] if args.check == "all" else [args.check]
            raise SystemExit(0 if check_board(names, clf, fz, args.retries) else 1)
        src = BoardSource(args.golden, clf, fz, pace=args.sim_pace, max_queue=args.max_queue,
                          retries=args.retries)
        lm.run_live(src, args.speed)
        src.close()
    finally:
        transport.close()


if __name__ == "__main__":
    main()
