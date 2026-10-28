#!/usr/bin/env python3
"""Minimal UART smoke test for the single-tile Conv1D FPGA accelerator.

Example (Windows):
    py comm/host/ecg_host.py COM4 --full

The test sends signed INT8 X=[1,-2,3,4,-1] and three length-3 filters,
then checks the nine INT32 results returned by RUN_TILE.
"""

from __future__ import annotations

import argparse
import random
import struct
import sys
import time
import re
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime

# Keep Vietnamese diagnostics readable in Windows consoles that still default
# to a legacy code page (for example cp1252).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")

try:
    import serial
except ImportError as exc:
    raise SystemExit(
        "Missing dependency pyserial. Install it with: py -m pip install pyserial"
    ) from exc


SOF = b"\xA5\x5A"

CMD_WRITE_IFMAP = 0x01
CMD_WRITE_WEIGHT = 0x02
CMD_CONFIG_TILE = 0x03
CMD_RUN_TILE = 0x04
CMD_WRITE_BIAS = 0x06
CMD_CONFIG_POST = 0x07

RSP_ACK = 0x80
RSP_NACK = 0x81
RSP_RESULT = 0x82


COMMAND_NAMES = {
    CMD_WRITE_IFMAP: "WRITE_IFMAP",
    CMD_WRITE_WEIGHT: "WRITE_WEIGHT",
    CMD_CONFIG_TILE: "CONFIG_TILE",
    CMD_RUN_TILE: "RUN_TILE",
    CMD_WRITE_BIAS: "WRITE_BIAS",
    CMD_CONFIG_POST: "CONFIG_POST",
    RSP_ACK: "ACK",
    RSP_NACK: "NACK",
    RSP_RESULT: "RESULT",
}


def crc16_ccitt_false(data: bytes) -> int:
    """CRC-16/CCITT-FALSE: poly 0x1021, init 0xffff, non-reflected."""
    crc = 0xFFFF
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


@dataclass(frozen=True)
class Packet:
    command: int
    sequence: int
    payload: bytes


class TeeStream:
    """Mirror terminal diagnostics to a UTF-8 log without hiding the terminal."""

    def __init__(self, *streams: object) -> None:
        self._streams = streams

    def write(self, text: str) -> int:
        for stream in self._streams:
            stream.write(text)
        return len(text)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()

    def isatty(self) -> bool:
        return False


class Accelerator:
    def __init__(self, port: str, baud: int, timeout: float, trace: bool = False) -> None:
        self.uart = serial.Serial(port=port, baudrate=baud, timeout=timeout)
        self.timeout = timeout
        self.sequence = 0
        self.trace = trace
        self.trace_start = time.monotonic()
        # Discard stale bytes left by a prior failed trial; this does not reset the FPGA.
        self.uart.reset_input_buffer()
        self.uart.reset_output_buffer()

    def close(self) -> None:
        self.uart.close()

    def _trace_packet(self, direction: str, packet: Packet) -> None:
        """Show the exact on-wire packet after CRC verification/building."""
        if not self.trace:
            return
        header = bytes((packet.command, packet.sequence)) + struct.pack("<H", len(packet.payload))
        wire = SOF + header + packet.payload + struct.pack("<H", crc16_ccitt_false(header + packet.payload))
        elapsed = time.monotonic() - self.trace_start
        name = COMMAND_NAMES.get(packet.command, f"0x{packet.command:02X}")
        print(
            f"[{elapsed:8.3f}s] {direction} {name:<13} "
            f"cmd=0x{packet.command:02X} seq={packet.sequence:03d} "
            f"payload={len(packet.payload):3d}B  {wire.hex(' ')}"
        )

    def _read_exact(self, count: int, deadline: float) -> bytes:
        received = bytearray()
        while len(received) < count:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"UART timeout: received {len(received)}/{count} bytes")
            chunk = self.uart.read(count - len(received))
            if chunk:
                received.extend(chunk)
        return bytes(received)

    def _read_packet(self) -> Packet:
        deadline = time.monotonic() + self.timeout
        previous = None
        while True:
            byte = self._read_exact(1, deadline)[0]
            if previous == SOF[0] and byte == SOF[1]:
                break
            previous = byte

        header = self._read_exact(4, deadline)
        command, sequence, payload_len = header[0], header[1], header[2] | (header[3] << 8)
        if payload_len > 256:
            raise RuntimeError(f"Invalid response length {payload_len}")
        payload = self._read_exact(payload_len, deadline)
        received_crc = struct.unpack("<H", self._read_exact(2, deadline))[0]
        calculated_crc = crc16_ccitt_false(header + payload)
        if received_crc != calculated_crc:
            raise RuntimeError(
                f"Bad response CRC: received 0x{received_crc:04X}, expected 0x{calculated_crc:04X}"
            )
        packet = Packet(command, sequence, payload)
        self._trace_packet("RX", packet)
        return packet

    def exchange(self, command: int, payload: bytes, expected_response: int) -> Packet:
        if len(payload) > 256:
            raise ValueError("Payload exceeds the RTL maximum of 256 bytes")
        sequence = self.sequence
        header = bytes((command, sequence)) + struct.pack("<H", len(payload))
        wire = SOF + header + payload + struct.pack("<H", crc16_ccitt_false(header + payload))
        self._trace_packet("TX", Packet(command, sequence, payload))
        self.uart.write(wire)
        self.uart.flush()

        response = self._read_packet()
        self.sequence = (self.sequence + 1) & 0xFF
        if response.sequence != sequence:
            raise RuntimeError(f"Sequence mismatch: got {response.sequence}, expected {sequence}")
        if response.command == RSP_NACK:
            original = response.payload[0] if response.payload else None
            error = response.payload[1] if len(response.payload) > 1 else None
            raise RuntimeError(f"FPGA NACK: original_cmd={original!r}, error={error!r}")
        if response.command != expected_response:
            raise RuntimeError(
                f"Unexpected response 0x{response.command:02X}; expected 0x{expected_response:02X}"
            )
        if expected_response == RSP_ACK:
            if len(response.payload) != 2 or response.payload[0] != command or response.payload[1] != 0:
                raise RuntimeError(f"Malformed ACK payload: {response.payload.hex(' ')}")
        return response

    def write_tensor(self, command: int, values: list[int], base_address: int = 0) -> None:
        if command not in (CMD_WRITE_IFMAP, CMD_WRITE_WEIGHT):
            raise ValueError("write_tensor command must be WRITE_IFMAP or WRITE_WEIGHT")
        if not 0 <= base_address <= 0xFFFF:
            raise ValueError("Base address must fit in u16")
        for value in values:
            if not -128 <= value <= 127:
                raise ValueError(f"INT8 value out of range: {value}")

        # The controller accepts at most 254 tensor bytes after the u16 address.
        for offset in range(0, len(values), 254):
            chunk = values[offset : offset + 254]
            payload = struct.pack("<H", base_address + offset) + bytes(value & 0xFF for value in chunk)
            self.exchange(command, payload, RSP_ACK)

    def write_bias(self, values: list[int], base_address: int = 0) -> None:
        if not 1 <= len(values) <= 3:
            raise ValueError("The current tile supports one to three INT32 biases")
        if not 0 <= base_address <= 2 or base_address + len(values) > 3:
            raise ValueError("Bias address range must stay within channels 0..2")
        payload = struct.pack("<H", base_address) + struct.pack("<" + "i" * len(values), *values)
        self.exchange(CMD_WRITE_BIAS, payload, RSP_ACK)

    def configure_post(self, shifts: list[int], relu: bool = True) -> None:
        if len(shifts) != 3:
            raise ValueError("CONFIG_POST cần đúng 3 shift, một cho mỗi cột PE")

        for shift in shifts:
            if not -128 <= shift <= 127:
                raise ValueError(f"Shift INT8 không hợp lệ: {shift}")
        # shift cột 0 | shift cột 1 | shift cột 2 | relu
        payload = bytes(value & 0xFF for value in shifts)
        payload += bytes((1 if relu else 0,))
        self.exchange(CMD_CONFIG_POST, payload, RSP_ACK)

    def configure(self, k_len: int, window_step: int, rows: int, cols: int,
                  x_base: int = 0, w_base: int = 0, preload_bias: bool = False) -> None:
        payload = struct.pack("<HHBBHH", k_len, window_step, rows, cols, x_base, w_base)
        if preload_bias:
            payload += b"\x01"
        self.exchange(CMD_CONFIG_TILE, payload, RSP_ACK)

    def run(self) -> tuple[list[int], list[int] | None]:
        response = self.exchange(CMD_RUN_TILE, b"", RSP_RESULT)
        if len(response.payload) == 36:
            raw = list(struct.unpack("<9i", response.payload))
            return raw, None

        if len(response.payload) == 45:
            raw = list(struct.unpack("<9i", response.payload[:36]))
            post = list(struct.unpack("<9b", response.payload[36:45]))
            return raw, post

        raise RuntimeError(
            f"Expected 36 raw INT32 bytes or 45 raw+post bytes; got {len(response.payload)}."
        )


def expected_tile(samples: list[int], weights: list[int], k_len: int, window_step: int,
                  rows: int, cols: int, bias: list[int] | None = None) -> list[int]:
    """Reference for the tile's physical 3x3 row-major INT32 output."""
    expected = [0] * 9
    for row in range(rows):
        for col in range(cols):
            value = 0 if bias is None else bias[col]
            for k in range(k_len):
                value += samples[row * window_step + k] * weights[col * k_len + k]
            expected[row * 3 + col] = value
    return expected


def execute_case(accelerator: Accelerator, name: str, samples: list[int], weights: list[int],
                 k_len: int, rows: int, cols: int, *, write_ifmap: bool = True,
                 write_weight: bool = True, preload_bias: list[int] | None = None,
                 verbose: bool = True) -> None:
    if verbose:
        print(f"\n=== {name} ===")
    if write_ifmap:
        if verbose:
            print(f"  1. WRITE_IFMAP: {len(samples)} signed INT8 samples -> ACK")
        accelerator.write_tensor(CMD_WRITE_IFMAP, samples)
    elif verbose:
        print("  1. WRITE_IFMAP: reuse active IFMAP bank")
    if write_weight:
        if verbose:
            print(f"  2. WRITE_WEIGHT: {len(weights)} signed INT8 weights -> ACK")
        accelerator.write_tensor(CMD_WRITE_WEIGHT, weights)
    elif verbose:
        print("  2. WRITE_WEIGHT: reuse active weight bank")
    if preload_bias is not None:
        if verbose:
            print(f"  3. WRITE_BIAS: {preload_bias} -> ACK")
        accelerator.write_bias(preload_bias)
    elif verbose:
        print("  3. WRITE_BIAS: disabled")
    if verbose:
        print(f"  4. CONFIG_TILE: K={k_len}, step=1, active={rows}x{cols}, preload={preload_bias is not None} -> ACK")

    accelerator.configure_post([0, 0, 0], relu=False)
    accelerator.configure(
        k_len=k_len,
        window_step=1,
        rows=rows,
        cols=cols,
        preload_bias=preload_bias is not None,
    )
    if verbose:
        print("  5. RUN_TILE -> RESULT")
    actual, _post = accelerator.run()
    expected = expected_tile(samples, weights, k_len, 1, rows, cols, preload_bias)
    if actual != expected:
        raise RuntimeError(f"{name}: expected {expected}, got {actual}")
    if verbose:
        print("  expected / actual (INT32):")
        for row in range(3):
            expected_row = " ".join(f"{value:5d}" for value in expected[row * 3 : row * 3 + 3])
            actual_row = " ".join(f"{value:5d}" for value in actual[row * 3 : row * 3 + 3])
            print(f"    [{expected_row}] / [{actual_row}]")
    print(f"PASS {name}")


def run_smoke_test(accelerator: Accelerator, verbose: bool = True) -> tuple[list[int], list[int]]:
    samples = [1, -2, 3, 4, -1]
    weights = [2, -1, 3, 1, 1, 1, -1, 2, 0]
    execute_case(accelerator, "baseline 3x3 K=3", samples, weights, 3, 3, 3, verbose=verbose)
    return samples, weights


def run_full_regression(accelerator: Accelerator, random_cases: int, verbose: bool = True) -> None:
    samples, weights = run_smoke_test(accelerator, verbose)

    # Verify bias preload in all three columns. Tensor banks are intentionally reused.
    execute_case(
        accelerator,
        "bias preload",
        samples,
        weights,
        3,
        3,
        3,
        write_ifmap=False,
        write_weight=False,
        preload_bias=[10, -3, 7],
        verbose=verbose,
    )

    # Legacy CONFIG_TILE disables preload again. First rerun checks active-bank reuse.
    execute_case(
        accelerator,
        "reuse active banks",
        samples,
        weights,
        3,
        3,
        3,
        write_ifmap=False,
        write_weight=False,
        verbose=verbose,
    )

    # Replace only the input staging bank. The active weight bank must be retained.
    new_samples = [2, 2, 2, 2, 2]
    execute_case(
        accelerator,
        "input-only ping-pong swap",
        new_samples,
        weights,
        3,
        3,
        3,
        write_weight=False,
        verbose=verbose,
    )

    # Inactive physical rows/columns must be zero in the returned 3x3 result.
    execute_case(
        accelerator,
        "partial tile 2x2",
        new_samples,
        weights,
        3,
        2,
        2,
        write_ifmap=False,
        write_weight=False,
        verbose=verbose,
    )

    # Deterministic signed INT8 random vectors exercise K limits and all active sizes.
    generator = random.Random(0xEC60)  # fixed seed: a failure is reproducible
    k_choices = (1, 3, 16)
    for case_index in range(random_cases):
        k_len = k_choices[case_index % len(k_choices)]
        rows = 1 + generator.randrange(3)
        cols = 1 + generator.randrange(3)
        samples = [generator.randrange(-32, 33) for _ in range(k_len + rows - 1)]
        weights = [generator.randrange(-32, 33) for _ in range(k_len * cols)]
        execute_case(
            accelerator,
            f"random {case_index + 1:02d}/{random_cases:02d} (K={k_len}, {rows}x{cols})",
            samples,
            weights,
            k_len,
            rows,
            cols,
            verbose=verbose,
        )


ROOT = Path(__file__).resolve().parents[4]
GOLDEN_DIR = ROOT / "export" / "golden"
WEIGHTS_HEADER = ROOT / "export" / "weights.h"
LOG_DIR = ROOT / "results" / "hardware_logs"

CONV0_LEN = 360
CONV0_COUT = 16
CONV0_K = 7
ARRAY_ROWS = 3
ARRAY_COLS = 3

def signed_int8_bytes(data: bytes) -> list[int]:
    return [byte if byte < 128 else byte - 256 for byte in data]


def c_array_values(header: str, c_type: str, name: str) -> list[int]:
    pattern = (
        rf"static\s+const\s+{re.escape(c_type)}\s+"
        rf"{re.escape(name)}\s*\[[^=]+\]\s*=\s*\{{(.*?)\}};"
    )
    match = re.search(pattern, header, flags=re.DOTALL)
    if match is None:
        raise RuntimeError(f"Không tìm thấy {name} trong weights.h")

    return [int(value) for value in re.findall(r"-?\d+", match.group(1))]


def load_conv0_parameters() -> tuple[list[int], list[int], list[int]]:
    header = WEIGHTS_HEADER.read_text(encoding="utf-8")

    weights = c_array_values(header, "int8_t", "CONV0_W")
    bias = c_array_values(header, "int32_t", "CONV0_B")
    shifts = c_array_values(header, "int8_t", "CONV0_SHIFT")

    if len(weights) != CONV0_COUT * CONV0_K:
        raise RuntimeError(f"CONV0_W cần 112 giá trị, nhận {len(weights)}")
    if len(bias) != CONV0_COUT:
        raise RuntimeError(f"CONV0_B cần 16 giá trị, nhận {len(bias)}")
    if len(shifts) != CONV0_COUT:
        raise RuntimeError(f"CONV0_SHIFT cần 16 giá trị, nhận {len(shifts)}")

    return weights, bias, shifts


def load_conv0_beat(record: str, beat_index: int) -> tuple[list[int], list[int]]:
    z_path = GOLDEN_DIR / f"{record}_beat_z_q.bin"
    golden_path = GOLDEN_DIR / f"{record}_beat_conv0.bin"

    z_blob = z_path.read_bytes()
    golden_blob = golden_path.read_bytes()

    input_offset = beat_index * CONV0_LEN
    input_end = input_offset + CONV0_LEN
    if input_end > len(z_blob):
        raise ValueError(f"beat={beat_index} vượt số beat trong {z_path.name}")

    golden_size = CONV0_COUT * CONV0_LEN
    golden_offset = beat_index * golden_size
    golden_end = golden_offset + golden_size
    if golden_end > len(golden_blob):
        raise ValueError(f"beat={beat_index} vượt số beat trong {golden_path.name}")

    samples = signed_int8_bytes(z_blob[input_offset:input_end])
    golden = signed_int8_bytes(golden_blob[golden_offset:golden_end])
    return samples, golden

def run_conv0_record(
    accelerator: Accelerator,
    record: str,
    beat_index: int,
) -> None:
    samples, golden = load_conv0_beat(record, beat_index)
    weights, bias, shifts = load_conv0_parameters()

    # CNN dùng same-padding với K=7, tức là thêm 3 số 0 ở mỗi đầu.
    padded_samples = [0, 0, 0] + samples + [0, 0, 0]

    print(f"Conv0: record={record}, beat={beat_index}")
    print("Upload IFMAP: 366 bytes, gồm zero-padding hai đầu")
    accelerator.write_tensor(CMD_WRITE_IFMAP, padded_samples)

    print("Upload weight: 16 filters x 7 = 112 bytes")
    accelerator.write_tensor(CMD_WRITE_WEIGHT, weights)

    result = [[0 for _ in range(CONV0_LEN)] for _ in range(CONV0_COUT)]
    post_mismatches: list[tuple[int, int, int, int, int, int, int]] = []
    raw_mismatches: list[tuple[int, int, int, int, int, int]] = []
    total_tiles = ((CONV0_LEN + ARRAY_ROWS - 1) // ARRAY_ROWS) * (
        (CONV0_COUT + ARRAY_COLS - 1) // ARRAY_COLS
    )
    tile_number = 0

    for c0 in range(0, CONV0_COUT, ARRAY_COLS):
        cols = min(ARRAY_COLS, CONV0_COUT - c0)

        # Bias memory của tile chỉ có 3 vị trí, nên nạp lại cho mỗi group channel.
        accelerator.write_bias(bias[c0:c0 + cols], base_address=0)

        # CONFIG_POST luôn mang 3 shift; lane không active pad bằng 0.
        group_shifts = shifts[c0:c0 + cols] + [0] * (ARRAY_COLS - cols)
        accelerator.configure_post(group_shifts, relu=True)

        print(f"  Channel {c0:02d}..{c0 + cols - 1:02d}: 120 tile")

        for t0 in range(0, CONV0_LEN, ARRAY_ROWS):
            rows = min(ARRAY_ROWS, CONV0_LEN - t0)
            tile_number += 1

            accelerator.configure(
                k_len=CONV0_K,
                window_step=1,
                rows=rows,
                cols=cols,
                x_base=t0,
                w_base=c0 * CONV0_K,
                preload_bias=True,
            )

            raw9, post9 = accelerator.run()
            if post9 is None:
                raise RuntimeError(
                    "Board đang trả raw-only. Hãy compile/nạp lại với POST_ENABLE=1."
                )

            if accelerator.trace:
                print(
                    f"[TILE {tile_number:03d}/{total_tiles}] "
                    f"time={t0:03d}..{t0 + rows - 1:03d}, "
                    f"channel={c0:02d}..{c0 + cols - 1:02d}, "
                    f"x_base={t0}, w_base={c0 * CONV0_K}, K={CONV0_K}"
                )

            for row in range(rows):
                for col in range(cols):
                    channel = c0 + col
                    time_index = t0 + row
                    actual = post9[row * ARRAY_COLS + col]
                    weight_base = channel * CONV0_K
                    mac = sum(
                        padded_samples[time_index + k] * weights[weight_base + k]
                        for k in range(CONV0_K)
                    )
                    expected_raw = mac + bias[channel]
                    actual_raw = raw9[row * ARRAY_COLS + col]

                    # Golden có layout [beat][channel][time].
                    expected = golden[channel * CONV0_LEN + time_index]
                    result[channel][time_index] = actual

                    if accelerator.trace:
                        print(
                            f"  t={time_index:03d} c={channel:02d}: "
                            f"MAC={mac:8d} + bias={bias[channel]:6d} "
                            f"= raw(ref/fpga) {expected_raw:8d}/{actual_raw:8d}; "
                            f"shift={shifts[channel]:3d}, post(golden/fpga)={expected:4d}/{actual:4d}"
                        )

                    if actual_raw != expected_raw and len(raw_mismatches) < 20:
                        raw_mismatches.append(
                            (channel, time_index, expected_raw, actual_raw, mac, bias[channel])
                        )

                    if actual != expected and len(post_mismatches) < 20:
                        post_mismatches.append(
                            (
                                channel,
                                time_index,
                                expected,
                                actual,
                                actual_raw,
                                bias[channel],
                                shifts[channel],
                            )
                        )

    if raw_mismatches or post_mismatches:
        if raw_mismatches:
            print(f"FAIL raw Conv0: in tối đa {len(raw_mismatches)} lỗi đầu:")
            for channel, time_index, expected, actual, mac, b in raw_mismatches:
                print(
                    f"  c={channel:02d}, t={time_index:03d}: "
                    f"MAC={mac:8d}, bias={b:6d}, raw_ref={expected:8d}, raw_fpga={actual:8d}"
                )
        if post_mismatches:
            print(f"FAIL post Conv0: in tối đa {len(post_mismatches)} lỗi đầu:")
            for channel, time_index, expected, actual, raw, b, shift in post_mismatches:
                print(
                    f"  c={channel:02d}, t={time_index:03d}: "
                    f"golden={expected:4d}, fpga={actual:4d}, "
                    f"raw={raw:8d}, bias={b:6d}, shift={shift}"
                )
        raise RuntimeError("Conv0 không khớp reference/golden")

    print("PASS Conv0: 5760 / 5760 giá trị khớp golden")

def main() -> int:
    parser = argparse.ArgumentParser(description="Run Conv1D hardware tests over UART.")
    parser.add_argument("port", help="Serial port, e.g. COM4")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--timeout", type=float, default=2.0, help="Response timeout in seconds")
    parser.add_argument("--full", action="store_true", help="Run bias, bank and randomized hardware regression")
    parser.add_argument("--random-cases", type=int, default=20, help="Number of random cases used by --full")
    parser.add_argument("--quiet", action="store_true", help="Only print each test's PASS/FAIL summary")
    parser.add_argument(
        "--trace",
        action="store_true",
        help="In every UART packet and every Conv0 MAC/bias/requantise calculation",
    )
    parser.add_argument(
        "--log",
        nargs="?",
        const="auto",
        metavar="FILE",
        help="Save terminal output under results/hardware_logs; omit FILE for a timestamped name",
    )

    parser.add_argument(
        "--conv0",
        action="store_true",
        help="Chạy đủ Conv0 của một beat ECG và so với golden",
    )
    parser.add_argument(
        "--record",
        default="rec101_93600",
        help="Prefix record trong export/golden",
    )
    parser.add_argument(
        "--beat",
        type=int,
        default=0,
        help="Chỉ số beat trong record",
    )

    args = parser.parse_args()
    if args.random_cases < 1:
        parser.error("--random-cases must be positive")

    log_file = None
    original_stdout, original_stderr = sys.stdout, sys.stderr
    if args.log:
        if args.log == "auto":
            mode = "conv0" if args.conv0 else "uart"
            filename = f"{mode}_{args.record}_beat{args.beat:03d}_{datetime.now():%Y%m%d-%H%M%S}.log"
            log_path = LOG_DIR / filename
        else:
            requested_path = Path(args.log)
            log_path = requested_path if requested_path.is_absolute() else LOG_DIR / requested_path
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = log_path.open("w", encoding="utf-8", buffering=1)
        sys.stdout = TeeStream(original_stdout, log_file)
        sys.stderr = TeeStream(original_stderr, log_file)
        print(f"LOG: {log_path}")

    accelerator: Accelerator | None = None

    try:
        accelerator = Accelerator(args.port, args.baud, args.timeout, trace=args.trace)
        if args.conv0:
            if args.full:
                parser.error("Chỉ chọn một trong --full hoặc --conv0")
            run_conv0_record(accelerator, args.record, args.beat)
        elif args.full:
            run_full_regression(
                accelerator,
                args.random_cases,
                verbose=not args.quiet,
            )
        else:
            run_smoke_test(accelerator, verbose=not args.quiet)
    finally:
        if accelerator is not None:
            accelerator.close()
        if log_file is not None:
            print("LOG: closed")
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            log_file.close()
    print("PASS: UART -> SRAM -> Conv1D -> UART hardware")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
