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
from dataclasses import dataclass

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
RSP_ACK = 0x80
RSP_NACK = 0x81
RSP_RESULT = 0x82


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


class Accelerator:
    def __init__(self, port: str, baud: int, timeout: float) -> None:
        self.uart = serial.Serial(port=port, baudrate=baud, timeout=timeout)
        self.timeout = timeout
        self.sequence = 0
        # Discard stale bytes left by a prior failed trial; this does not reset the FPGA.
        self.uart.reset_input_buffer()
        self.uart.reset_output_buffer()

    def close(self) -> None:
        self.uart.close()

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
        return Packet(command, sequence, payload)

    def exchange(self, command: int, payload: bytes, expected_response: int) -> Packet:
        if len(payload) > 256:
            raise ValueError("Payload exceeds the RTL maximum of 256 bytes")
        sequence = self.sequence
        header = bytes((command, sequence)) + struct.pack("<H", len(payload))
        wire = SOF + header + payload + struct.pack("<H", crc16_ccitt_false(header + payload))
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

    def configure(self, k_len: int, window_step: int, rows: int, cols: int,
                  x_base: int = 0, w_base: int = 0, preload_bias: bool = False) -> None:
        payload = struct.pack("<HHBBHH", k_len, window_step, rows, cols, x_base, w_base)
        if preload_bias:
            payload += b"\x01"
        self.exchange(CMD_CONFIG_TILE, payload, RSP_ACK)

    def run(self) -> list[int]:
        response = self.exchange(CMD_RUN_TILE, b"", RSP_RESULT)
        if len(response.payload) != 36:
            raise RuntimeError(
                f"Expected 36 raw INT32 result bytes; got {len(response.payload)}. "
                "Recompile with POST_ENABLE=0 or update this host for post results."
            )
        return list(struct.unpack("<9i", response.payload))


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
    accelerator.configure(
        k_len=k_len,
        window_step=1,
        rows=rows,
        cols=cols,
        preload_bias=preload_bias is not None,
    )
    if verbose:
        print("  5. RUN_TILE -> RESULT")
    actual = accelerator.run()
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Conv1D hardware tests over UART.")
    parser.add_argument("port", help="Serial port, e.g. COM4")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--timeout", type=float, default=2.0, help="Response timeout in seconds")
    parser.add_argument("--full", action="store_true", help="Run bias, bank and randomized hardware regression")
    parser.add_argument("--random-cases", type=int, default=20, help="Number of random cases used by --full")
    parser.add_argument("--quiet", action="store_true", help="Only print each test's PASS/FAIL summary")
    args = parser.parse_args()
    if args.random_cases < 1:
        parser.error("--random-cases must be positive")

    accelerator = Accelerator(args.port, args.baud, args.timeout)
    try:
        if args.full:
            run_full_regression(accelerator, args.random_cases, verbose=not args.quiet)
        else:
            run_smoke_test(accelerator, verbose=not args.quiet)
    finally:
        accelerator.close()
    print("PASS: UART -> SRAM -> Conv1D -> UART hardware regression")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
