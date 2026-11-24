"""Stop-and-wait host for rtl/cnn1d/top/layer_cmd_controller.v.

This client is intentionally separate from TileHost: the layer protocol sends
one set of layer data/configuration and asks the FPGA to execute the layer's
internal tiles. It never issues the old RUN_TILE command and never retries a
command automatically.
"""
from __future__ import annotations

import struct
import time
from dataclasses import dataclass, field
from typing import Iterable

from python.link import proto_layer as pl


class LayerLinkError(RuntimeError):
    def __init__(self, kind: str, message: str, cmd: int | None = None, seq: int | None = None):
        super().__init__(message)
        self.kind = kind
        self.cmd = cmd
        self.seq = seq


@dataclass
class LayerHostStats:
    transactions: int = 0
    tx_bytes: int = 0
    rx_bytes: int = 0
    noise_bytes: int = 0
    by_cmd: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class RunResult:
    status: int
    cycles: int


class LayerHost:
    """Layer command client over any transport exposing write/read/clock.

    `transport.read(timeout)` must return available bytes (or b'' on timeout).
    `transport.clock()` must return a monotonic time. SerialTransport and
    RawLog(SerialTransport, ...) satisfy this interface.
    """

    def __init__(self, transport, timeout: float = 3.0, max_payload: int = pl.MAX_PAYLOAD):
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self.transport = transport
        self.timeout = float(timeout)
        self.seq = 0
        self.decoder = pl.Decoder(max_payload=max_payload)
        self.max_payload = int(max_payload)
        self.info_data: pl.HardwareInfo | None = None
        self.stats = LayerHostStats()

    def _clock(self) -> float:
        clock = getattr(self.transport, "clock", None)
        return clock() if clock else time.monotonic()

    def _exchange(self, cmd: int, payload: bytes, expect: int) -> bytes:
        name = pl.CMD_NAMES.get(cmd, f"0x{cmd:02X}")
        try:
            packet = pl.frame(cmd, self.seq, payload, self.max_payload)
        except ValueError as exc:
            raise LayerLinkError("argument", str(exc), cmd, self.seq) from exc
        seq = self.seq
        self.seq = (self.seq + 1) & 0xFF

        self.transport.write(packet)
        self.stats.transactions += 1
        self.stats.tx_bytes += len(packet)
        self.stats.by_cmd[name] = self.stats.by_cmd.get(name, 0) + 1
        deadline = self._clock() + self.timeout

        while True:
            events = self.decoder.feed(b"")
            for event in events:
                answer = self._check_event(event, cmd, seq, expect)
                if answer is not None:
                    return answer

            left = deadline - self._clock()
            if left <= 0:
                want = pl.RSP_NAMES[expect]
                raise LayerLinkError(
                    "timeout",
                    f"timeout waiting {want} for {name} (SEQ {seq}) after {self.timeout:.2f}s; "
                    "check COM port, UART wiring/GND, baud, loaded bitstream and reset; command was not retried",
                    cmd,
                    seq,
                )
            data = self.transport.read(left)
            if not data:
                continue
            self.stats.rx_bytes += len(data)
            for event in self.decoder.feed(data):
                answer = self._check_event(event, cmd, seq, expect)
                if answer is not None:
                    return answer

    def _check_event(self, event, cmd: int, seq: int, expect: int) -> bytes | None:
        name = pl.CMD_NAMES.get(cmd, f"0x{cmd:02X}")
        if isinstance(event, pl.BadFrame):
            if event.reason.startswith("noise"):
                self.stats.noise_bytes += len(event.data)
                return None
            raise LayerLinkError("corrupt", f"corrupt FPGA response to {name} (SEQ {seq}): {event.reason}; no retry", cmd, seq)
        if event.seq != seq:
            raise LayerLinkError(
                "sequence",
                f"response SEQ {event.seq} does not match request SEQ {seq} for {name}; no retry",
                cmd,
                seq,
            )
        if event.cmd == pl.RSP_NACK:
            if len(event.payload) != 2:
                raise LayerLinkError("format", f"malformed NACK payload for {name}: {event.payload.hex()}", cmd, seq)
            original, code = event.payload
            text = pl.NACK_TEXT.get(code, "unknown NACK code")
            raise LayerLinkError("nack", f"FPGA NACK {code} ({text}) for CMD 0x{original:02X} {pl.CMD_NAMES.get(original, '')}", cmd, seq)
        if event.cmd != expect:
            raise LayerLinkError(
                "format",
                f"expected {pl.RSP_NAMES.get(expect, hex(expect))} for {name}, got {pl.RSP_NAMES.get(event.cmd, hex(event.cmd))}",
                cmd,
                seq,
            )
        if expect == pl.RSP_ACK:
            if len(event.payload) != 2 or event.payload[0] != cmd or event.payload[1] != 0:
                raise LayerLinkError("format", f"invalid ACK payload for {name}: {event.payload.hex()}", cmd, seq)
        return event.payload

    def info(self) -> pl.HardwareInfo:
        payload = self._exchange(pl.CMD_INFO, b"", pl.RSP_RESULT)
        try:
            info = pl.decode_info(payload)
        except ValueError as exc:
            raise LayerLinkError("format", f"invalid INFO response: {exc}", pl.CMD_INFO) from exc
        self.info_data = info
        self.max_payload = info.max_payload
        self.decoder.set_max_payload(info.max_payload)
        return info

    def _require_info(self) -> pl.HardwareInfo:
        if self.info_data is None:
            raise LayerLinkError("state", "call info() first so host uses the FPGA-reported MAX_PAYLOAD")
        return self.info_data

    def write_bytes(self, cmd: int, address: int, values: Iterable[int]) -> int:
        info = self._require_info()
        if cmd not in (pl.CMD_WRITE_X, pl.CMD_WRITE_W, pl.CMD_WRITE_SHIFT):
            raise ValueError("write_bytes only accepts WRITE_X, WRITE_W or WRITE_SHIFT")
        # Keep signed values here: encode_write_bytes() does the int8 -> byte conversion itself.
        # (Converting to bytes first and passing those on made every negative value fail.)
        vals = [int(v) for v in values]
        if not vals:
            raise ValueError("write data must not be empty")
        pl._int8_bytes(vals)  # range check once, before anything is sent
        max_chunk = info.max_payload - 2
        if max_chunk < 1:
            raise LayerLinkError("configuration", "MAX_PAYLOAD must be at least 3 for INT8 writes", cmd)
        if address < 0 or address + len(vals) > 0x10000:
            raise ValueError("write address range must fit in u16")
        for offset in range(0, len(vals), max_chunk):
            part = vals[offset:offset + max_chunk]
            body = pl.encode_write_bytes(cmd, address + offset, part)
            self._exchange(cmd, body, pl.RSP_ACK)
        return len(vals)

    def write_x(self, address: int, values: Iterable[int]) -> int:
        info = self._require_info()
        vals = tuple(values)
        if address < 0 or address + len(vals) > info.ifmap_depth:
            raise ValueError("WRITE_X exceeds FPGA IFMAP_MEM_DEPTH")
        return self.write_bytes(pl.CMD_WRITE_X, address, vals)

    def write_w(self, address: int, values: Iterable[int]) -> int:
        info = self._require_info()
        vals = tuple(values)
        if address < 0 or address + len(vals) > info.wgt_depth:
            raise ValueError("WRITE_W exceeds FPGA WGT_MEM_DEPTH")
        return self.write_bytes(pl.CMD_WRITE_W, address, vals)

    def write_shift(self, address: int, values: Iterable[int]) -> int:
        info = self._require_info()
        vals = tuple(values)
        if address < 0 or address + len(vals) > 2 * info.cout_max:
            raise ValueError("WRITE_SHIFT exceeds 2*COUT_MAX")
        return self.write_bytes(pl.CMD_WRITE_SHIFT, address, vals)

    def write_bias(self, base: int, values: Iterable[int]) -> int:
        info = self._require_info()
        vals = tuple(int(v) for v in values)
        if not vals:
            raise ValueError("WRITE_BIAS requires at least one bias")
        limit = pl.BIAS_DEPTH if info.supports_resident else info.cout_max     # bias RAM is deeper than COUT_MAX only with b_base
        if base < 0 or base + len(vals) > limit:
            raise ValueError(f"WRITE_BIAS exceeds {'BIAS_DEPTH' if info.supports_resident else 'COUT_MAX'} ({limit})")
        per_packet = min(32, (info.max_payload - 2) // 4)
        if per_packet < 1:
            raise LayerLinkError("configuration", "MAX_PAYLOAD too small for one INT32 bias", pl.CMD_WRITE_BIAS)
        for offset in range(0, len(vals), per_packet):
            part = vals[offset:offset + per_packet]
            self._exchange(pl.CMD_WRITE_BIAS, pl.encode_write_bias(base + offset, part), pl.RSP_ACK)
        return len(vals)

    def swap(self, x: bool = False, w: bool = False) -> None:
        self._exchange(pl.CMD_SWAP, pl.encode_swap(x=x, w=w), pl.RSP_ACK)

    def config(self, out_len: int, cout: int, k_len: int, window_step: int,
               pool: bool = False, gap: bool = False, pad_next: int = 0,
               flags: int | None = None, w_base: int | None = None, b_base: int | None = None) -> None:
        info = self._require_info()
        if cout > info.cout_max or k_len > info.feeder_depth:
            raise ValueError("CONFIG cout/K exceed FPGA-reported limits")
        if (w_base is not None or b_base is not None) and not info.supports_resident:
            raise ValueError("this bitstream has no w_base/b_base (INFO features bit 4 clear)")
        if w_base is not None and w_base + cout * k_len > info.wgt_depth:
            raise ValueError("CONFIG w_base + Cout*K exceeds the weight SRAM")
        if b_base is not None and b_base + cout > pl.BIAS_DEPTH:
            raise ValueError("CONFIG b_base + Cout exceeds BIAS_DEPTH")
        self._exchange(
            pl.CMD_CONFIG,
            pl.encode_config(out_len, cout, k_len, window_step, pool, gap, pad_next, flags, w_base, b_base),
            pl.RSP_ACK,
        )

    def run(self) -> RunResult:
        payload = self._exchange(pl.CMD_RUN, b"", pl.RSP_RESULT)
        if len(payload) != 5:
            raise LayerLinkError("format", f"RUN RESULT must be 5 B [status:u8, cycles:u32], got {len(payload)} B", pl.CMD_RUN)
        # Keep the hardware status visible to the layer runner. Huy's runner still
        # reads the output after a non-zero status so it can report both overrun
        # status and any resulting golden-data mismatch.
        return RunResult(payload[0], struct.unpack_from("<I", payload, 1)[0])

    def read_fmap(self, offset: int, length: int) -> bytes:
        info = self._require_info()
        if offset < 0 or length < 0 or offset + length > info.ifmap_depth:
            raise ValueError("READ_FMAP range exceeds IFMAP_MEM_DEPTH")
        if length == 0:
            return b""
        chunk_max = min(info.max_payload, 250)
        result = bytearray()
        for pos in range(0, length, chunk_max):
            count = min(chunk_max, length - pos)
            payload = self._exchange(pl.CMD_READ_FMAP, pl.encode_read_fmap(offset + pos, count), pl.RSP_RESULT)
            if len(payload) != count:
                raise LayerLinkError("format", f"READ_FMAP at {offset + pos}: expected {count} B, got {len(payload)} B", pl.CMD_READ_FMAP)
            result.extend(payload)
        return bytes(result)

    def read_gap(self) -> tuple[int, ...]:
        info = self._require_info()
        payload = self._exchange(pl.CMD_READ_GAP, b"", pl.RSP_RESULT)
        expected = info.cout_max * 2
        if len(payload) != expected:
            raise LayerLinkError("format", f"READ_GAP expected {expected} B, received {len(payload)} B", pl.CMD_READ_GAP)
        return struct.unpack(f"<{info.cout_max}H", payload)

    def smoke(self) -> dict:
        """Run one known 1-position x 1-channel Conv1D layer; no ECG model files needed.

        Inputs [1,2,3] and weights [1,1,1], bias=0, shift=0 produce output 6.
        The board is expected to have a layer-protocol bitstream and support one
        output position/channel. The result is read back from the output feature map.
        """
        info = self.info_data or self.info()
        if info.max_payload < 6:
            raise LayerLinkError("configuration", "MAX_PAYLOAD too small for one INT32 bias write")
        self.write_x(0, (1, 2, 3))
        self.swap(x=True)
        self.write_w(0, (1, 1, 1))
        self.swap(w=True)
        self.write_bias(0, (0,))
        self.write_shift(0, (0,))
        self.config(out_len=1, cout=1, k_len=3, window_step=1, pool=False, gap=False, pad_next=0)
        run = self.run()
        got = self.read_fmap(0, 1)
        value = struct.unpack("b", got)[0]
        return {
            "info": info,
            "run": run,
            "got": value,
            "expected": 6,
            "passed": run.status == 0 and value == 6,
            "sequence": [pl.CMD_INFO, pl.CMD_WRITE_X, pl.CMD_SWAP, pl.CMD_WRITE_W,
                         pl.CMD_SWAP, pl.CMD_WRITE_BIAS, pl.CMD_WRITE_SHIFT, pl.CMD_CONFIG,
                         pl.CMD_RUN, pl.CMD_READ_FMAP],
        }

    def close(self) -> None:
        close = getattr(self.transport, "close", None)
        if close:
            close()
