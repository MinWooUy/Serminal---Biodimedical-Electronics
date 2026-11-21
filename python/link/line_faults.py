"""Line timing and seeded fault injection shared by the in-process transports (fpga_sim.InProcessLink, tile_fpga_sim.TileLink)."""

from dataclasses import dataclass

import numpy as np

BYTE_TIME_S = 10 / 115200                 # 8N1: start + 8 data + stop bits


@dataclass
class LineFaults:
    """Per-frame fault probabilities on one direction of the line (seeded)."""
    flip_bit: float = 0.0          # flip one random bit
    drop_byte: float = 0.0         # delete one random byte and append 1..11 random bytes
    lose_frame: float = 0.0        # the whole frame never arrives

    def apply(self, frame: bytes, rng: np.random.Generator, counts: dict) -> bytes:
        r = rng.random()
        if r < self.lose_frame:
            counts["lost"] += 1
            return b""
        r -= self.lose_frame
        b = bytearray(frame)
        if r < self.flip_bit:
            i = int(rng.integers(0, len(b) * 8))
            b[i // 8] ^= 1 << (i % 8)
            counts["flipped"] += 1
            return bytes(b)
        r -= self.flip_bit
        if r < self.drop_byte:
            del b[int(rng.integers(0, len(b)))]
            b += rng.integers(0, 256, size=int(rng.integers(1, 12))).astype(np.uint8).tobytes()
            counts["dropped"] += 1
        return bytes(b)
