"""Stop-and-wait host of protocol v2 (docs/proto_v2.md, section 6).

PING -> wait PONG 500 ms; BEAT_IN -> wait RESULT 1 s; on timeout or NACK resend immediately with the same
beat_id, at most MAX_RETRIES (= 3) resends after the first send; a RESULT with another beat_id is ignored.
Works over any transport with write(bytes), read(timeout) -> bytes and clock() (python/link/fpga_sim.py).
"""

from dataclasses import dataclass, field

try:
    from python.link import proto_v2 as pv2
except ImportError:                       # stand-alone copy in the FPGA hand-off package
    import proto_v2 as pv2


RX_HEAD = 256


class LinkError(RuntimeError):
    pass


@dataclass
class HostStats:
    sent: int = 0                  # frames written (first sends + resends)
    resends: int = 0
    timeouts: int = 0
    nacks: dict = field(default_factory=lambda: {1: 0, 2: 0, 3: 0, 4: 0})
    ignored: int = 0               # well-formed packets that did not answer the pending request
    failed: int = 0                # requests abandoned after MAX_RETRIES resends

    def as_dict(self) -> dict:
        return {"sent": self.sent, "resends": self.resends, "timeouts": self.timeouts,
                "nacks": {f"reason_{k}": v for k, v in self.nacks.items()}, "ignored": self.ignored,
                "failed": self.failed}


class Host:
    def __init__(self, transport, pong_timeout: float = pv2.PONG_TIMEOUT_S,
                 result_timeout: float = pv2.RESULT_TIMEOUT_S, max_retries: int = pv2.MAX_RETRIES):
        self.t = transport
        self.pong_timeout, self.result_timeout, self.max_retries = pong_timeout, result_timeout, max_retries
        self.dec = pv2.Decoder(accept=pv2.FPGA_TO_HOST)
        self.stats = HostStats()
        self._pending = []                    # decoded packets not consumed yet
        self.rx_bytes = 0                     # every byte received (for diagnostics)
        self.rx_head = bytearray()            # the first RX_HEAD bytes received

    def _next_packet(self, deadline: float):
        """Next decoded packet before `deadline` (transport clock), or None on timeout."""
        while True:
            while self._pending:
                ev = self._pending.pop(0)
                if not isinstance(ev, pv2.FrameError):       # damaged frames are counted by the decoder
                    return ev
            left = deadline - self.t.clock()
            if left <= 0:
                return None
            data = self.t.read(left)
            if data:
                self.rx_bytes += len(data)
                if len(self.rx_head) < RX_HEAD:
                    self.rx_head += data[:RX_HEAD - len(self.rx_head)]
                self._pending += self.dec.feed(data)

    def request(self, pkt, is_answer, timeout: float):
        """Send pkt and wait for a packet with is_answer(p) True. Returns (answer, number of sends)."""
        frame = pv2.encode(pkt)
        for attempt in range(self.max_retries + 1):
            if attempt:
                self.stats.resends += 1
            self.t.write(frame)
            self.stats.sent += 1
            deadline = self.t.clock() + timeout
            while True:
                p = self._next_packet(deadline)
                if p is None:
                    self.stats.timeouts += 1
                    break
                if isinstance(p, pv2.Nack):
                    self.stats.nacks[p.reason] = self.stats.nacks.get(p.reason, 0) + 1
                    break
                if is_answer(p):
                    return p, attempt + 1
                self.stats.ignored += 1
        self.stats.failed += 1
        raise LinkError(f"no answer to {type(pkt).__name__} after {self.max_retries + 1} sends")

    def ping(self) -> pv2.Pong:
        return self.request(pv2.Ping(), lambda p: isinstance(p, pv2.Pong), self.pong_timeout)[0]

    def classify(self, beat_id: int, z_q, rr_q) -> tuple[pv2.Result, int]:
        pkt = pv2.BeatIn(int(beat_id), tuple(int(v) for v in z_q), tuple(int(v) for v in rr_q))
        return self.request(pkt, lambda p: isinstance(p, pv2.Result) and p.beat_id == pkt.beat_id,
                            self.result_timeout)
