# UART protocol: device → host monitor

Reference implementation: `python/viz/uart.py` (encoder + decoder). The simulated device
(`scripts/live_monitor.py --golden …`) and the board firmware must produce the **same bytes** for the same
input; the host (`live_monitor.py`) only ever consumes the decoded byte stream.

## Physical layer

8N1, default 115 200 baud. Payload budget at 360 Hz: SAMPLES frames of 8 samples = 59 B per 22.2 ms
≈ 2.66 kB/s, plus BEAT (39 B per beat) and STATS (34 B per second) ≈ 2.75 kB/s total, 24 % of 115 200 baud.

## Frame

| offset | size | field |
|---|---|---|
| 0 | 1 | `0xA5` |
| 1 | 1 | `0x5A` |
| 2 | 1 | `type` |
| 3 | 1 | `len` = payload length (0..255) |
| 4 | `len` | payload |
| 4 + `len` | 2 | `crc` = CRC-16 over `type`, `len`, payload (the sync word is not covered), **little-endian** |

All multi-byte fields are **little-endian** (the CRC included); signed fields are two's complement.

**CRC-16/CCITT-FALSE**: polynomial x¹⁶ + x¹² + x⁵ + 1 (0x1021), init 0xFFFF, MSB first (no input/output
reflection), xorout 0x0000. Check value: CRC-16(`"123456789"`) = **0x29B1**. Table-driven reference:
`uart.crc16` (tested equal to the bit-by-bit form below).

```c
uint16_t crc16_ccitt_false(const uint8_t *p, size_t n) {
    uint16_t c = 0xFFFF;
    while (n--) {
        c ^= (uint16_t)(*p++) << 8;
        for (int i = 0; i < 8; i++) c = (c & 0x8000) ? (uint16_t)((c << 1) ^ 0x1021) : (uint16_t)(c << 1);
    }
    return c;
}
/* frame trailer: buf[4 + len] = crc & 0xFF; buf[5 + len] = crc >> 8; */
```

## Frame types

### `0x01` SAMPLES — `len` = 5 + 6·n, 1 ≤ n ≤ 16

| field | type | meaning |
|---|---|---|
| `first_idx` | u32 | sample index (from power-on / segment start, 360 Hz) of the first sample |
| `n` | u8 | number of samples |
| `adc[n]` | i16 × n | raw ADC counts |
| `filt[n]` | i32 × n | 0.5–40 Hz integer filter output, ADC counts × 2^`filt_frac` (relative to adc_zero) |

The device sends samples in order; a jump of `first_idx` tells the host that samples were lost.
The simulated device sends n = 8 and closes a frame early before every BEAT or STATS frame, so that
all samples up to the emission time precede the beat.

### `0x02` BEAT — `len` = 33

| field | type | meaning |
|---|---|---|
| `r_idx` | u32 | sample index of the refined R peak (integer abs refinement) |
| `emit_idx` | u32 | sample index at which the beat is classified (latency = `emit_idx − r_idx`), see below |
| `pred` | u8 | argmax of the logits: 0 N, 1 S, 2 V, 3 F |
| `flags` | u8 | bit 0: detection from search-back; bit 1: refinement hit the window edge (coarse R kept) |
| `logits[4]` | i32 × 4 | FC2 logits, common scale 2^−`logit_frac` (the host computes softmax; probabilities are **not** sent) |
| `rr_prev` | u16 | RR before the beat (samples, clipped to 72..1080) |
| `rr_next` | u16 | RR after the beat (samples, clipped) |
| `rr_sum8` | u16 | sum of the last ≤ 8 RR including `rr_prev` (samples) |
| `rr_n8` | u8 | number of RR in `rr_sum8` (1..8) |

A beat is classified when the next R has been detected and its 360-sample window is complete, so the
RR features are exactly those of the offline pipeline. `emit_idx` is therefore deterministic:
`emit_idx = max(sample at which the next detection is emitted, r_idx + delay + 180 − 1)` (delay = 2). It is
part of the payload and of `beat_digest`, so the firmware must use exactly this rule.

### `0x03` STATS — `len` = 28, sent once per second (every 360 samples) and once after the last sample

| field | type | meaning |
|---|---|---|
| `sample_idx` | u32 | index of the last sample processed |
| `n_beats` | u32 | beats classified so far |
| `cycles_last` | u32 | CPU cycles of the last CNN inference |
| `cycles_max` | u32 | maximum CPU cycles of one inference so far |
| `cpu_hz` | u32 | CPU clock (Hz); **0 = unknown** (PC simulation: the cycle fields are 0 and meaningless) |
| `adc_gain` | u16 | ADC counts per mV (MIT-BIH: 200) |
| `filt_frac` | u8 | fractional bits of `filt` (12) |
| `logit_frac` | u8 | fractional bits of the logits (frozen Lite model: from `lite_int8.npz`, fc2 k + a_in) |
| `beat_digest` | u32 | **CRC-32/IEEE** accumulated over the payloads (33 bytes each, in order) of ALL BEAT frames since the start of the run; 0 before the first beat |

**CRC-32/IEEE** = `zlib.crc32` (reflected poly 0xEDB88320, init 0xFFFFFFFF, xorout 0xFFFFFFFF), chained
over the beat payloads: `digest = crc32(payload_k, digest)`. Check value: CRC-32(`"123456789"`) =
**0xCBF43926**. The device computes it on the payload bytes it sends; the host never needs it to decode.

A host that connects mid-stream learns the scales from the first STATS frame (≤ 1 s); before that it
shows predictions without probabilities. The **final STATS** (`sample_idx` = last sample) carries the
digest of the whole run.

## Receiver (resynchronisation and host checks)

The protocol above is all the device implements. The host decoder (`HostDecoder` in `python/viz/uart.py`)
accepts a frame only if ALL of the following hold, otherwise it discards **one** byte and scans again for
`0xA5 0x5A`:

1. valid header: known `type` and a `len` valid for it (else `bad_header`);
2. CRC-16 match (else `bad_crc`);
3. **host checks** (BEAT only): `pred` ∈ {0..3} and `start_idx ≤ r_idx < next_idx` (R inside the received
   sample range); a BEAT failing them is counted in **`transport_error`**.

SAMPLES frames that pass the CRC are **always accepted**: a discontinuous `first_idx` is recorded (a forward
gap adds its length to **`samples_lost`**; every discontinuity counts in `resyncs`) and the host
re-synchronises to the new `first_idx` (a backward jump, e.g. a device reset, also restarts the received
range). Frames are accepted as soon as they are complete (no lookahead on the next frame).
Counters: `frames_ok`, `bad_header`, `bad_crc`, `transport_error`, `skipped_bytes`, `samples_lost`,
`resyncs`. None of this changes the bytes on the line.

History (2026-10-16): an earlier host also required the next 2 bytes to be `0xA5 0x5A` (lookahead) and
rejected SAMPLES gaps > 2 s; both were removed. With CRC-16 they caught no false accept in any measurement
and only discarded intact frames (40 frames in `--check all --corrupt 0.02`).

Every single-bit error in a frame is rejected (tested exhaustively on every bit of each frame type; a flip
in `len` can never produce another valid SAMPLES length, since 2^k is never a multiple of 6). A lost byte is
a length change, which no CRC detects with certainty: the damaged frame can borrow bytes of what follows
and still match the CRC (probability ≈ 2^−16 per such frame with CRC-16).

Measurement (`tests/test_uart.py::test_no_false_accepts_2500_dropped_byte_frames`): 100 random streams
(seeds 4–103) of 300 frames, 25 damaged frames in each = 2 500 damaged frames out of 30 000;
damage A = one byte dropped; damage B = one byte dropped and 1–11 random bytes inserted after the frame
(the damage of `live_monitor --corrupt`).

| decoder | A: false accepts | A: intact frames also lost | B: false accepts | B: intact frames also lost |
|---|---|---|---|---|
| CRC-16 only (`FrameDecoder`) | 0 | 0 | 0 | 0 |
| **CRC-16 + host checks (`HostDecoder`, used)** | **0** | 44 | **0** | 44 |

The 44 intact frames are BEATs of the random test streams whose R lay in a SAMPLES frame that had just been
lost (R beyond the received range). On the real stream R is ≥ 181 samples before `emit_idx`, so this
almost never happens (0 in `--check all --corrupt 0.02`, see below).

With the earlier CRC-8 (1-byte check), CRC alone falsely accepted 9 / 2 500 frames of damage A, and one
false accept of damage B was seen in `--check all --corrupt 0.02` (rec209: a frame that lost a byte
borrowed one inserted garbage byte and the CRC-8 matched). With CRC-16 no false accept occurred in any
measurement. Losing frames is visible (counters, `samples_lost`, digest); a false accept would not be.

## Verification and GATE 4

**GATE 4 (bit-exactness of the device)**: on a **clean line**, for a whole record / golden segment, the
device's final `beat_digest` (and `n_beats`) equals the golden digest. The golden digest is computed
offline from the golden vectors (`python/viz/golden.py::golden_beats`: R, search-back flag, RR parts,
logits, prediction from `export/golden/*.bin`; `emit_idx` and the edge flag derived offline from the
shared detector and the golden filter output).

The per-frame comparison is kept for **diagnosis**, with two error classes:
- **transport errors**: frames rejected by the decoder, samples never received, golden beats never received;
- **computation errors**: ACCEPTED frames that differ from golden (SAMPLES: ADC / filter output; BEAT: the
  whole payload) or do not exist in golden.

Runs with `--corrupt` are **robustness checks**, not a gate: they report frames rejected, frames falsely
accepted (accepted frames that differ from golden) and whether the digests still match.

### Results (simulated device = `IntegerPipelineStream`, 2026-10-16)

`python scripts/live_monitor.py --check all` (clean line) — **GATE 4: PASS**

| segment | beats | golden digest | board digest | host-recomputed | computation errors | transport errors |
|---|---|---|---|---|---|---|
| rec101_93600 | 124/124 | 0xC0339239 | 0xC0339239 | 0xC0339239 | 0 | 0 |
| rec118_237600 | 149/149 | 0xCCA3485B | 0xCCA3485B | 0xCCA3485B | 0 | 0 |
| rec119_489600 | 129/129 | 0x50EEB298 | 0x50EEB298 | 0x50EEB298 | 0 | 0 |
| rec201_522000 | 115/115 | 0x7939B65D | 0x7939B65D | 0x7939B65D | 0 | 0 |
| rec207_604800 | 169/169 | 0x597D7339 | 0x597D7339 | 0x597D7339 | 0 | 0 |
| rec208_237600 | 191/191 | 0xDA42AE38 | 0xDA42AE38 | 0xDA42AE38 | 0 | 0 |
| rec209_198000 | 239/239 | 0xC20C9B2D | 0xC20C9B2D | 0xC20C9B2D | 0 | 0 |
| rec223_187200 | 190/190 | 0x7D2A6890 | 0x7D2A6890 | 0x7D2A6890 | 0 | 0 |

8/8 segments, 1 306 beats: board digest = golden, every sample and beat bit-exact, no transport error.

`python scripts/live_monitor.py --check all --corrupt 0.02` (robustness, not a gate; corruption seed 0;
host without lookahead, SAMPLES gaps accepted)

| segment | beats received | frames rejected | false accepts | missing samples | board digest = golden | host-recomputed = golden |
|---|---|---|---|---|---|---|
| rec101_93600 | 122/124 | 118 | 0 | 931 | yes | no (2 beats lost) |
| rec118_237600 | 146/149 | 118 | 0 | 920 | yes | no (3 beats lost) |
| rec119_489600 | 128/129 | 118 | 0 | 957 | yes | no (1 beat lost) |
| rec201_522000 | 115/115 | 117 | 0 | 943 | yes | yes |
| rec207_604800 | 169/169 | 119 | 0 | 964 | yes | yes |
| rec208_237600 | 191/191 | 120 | 0 | 987 | yes | yes |
| rec209_198000 | 236/239 | 121 | 0 | 920 | yes | no (3 beats lost) |
| rec223_187200 | 188/190 | 120 | 0 | 953 | yes | no (2 beats lost) |

Total: 951 frames rejected (33 bad header, 918 bad CRC, 0 transport_error), **0 false accepts**,
7 575 samples and 11 beats not received; board digest (final STATS) = golden on 8/8, host-recomputed digest
on the 3 segments without lost beats — every mismatch is explained by recorded transport losses.
Compared with the previous host (lookahead + gap rejection), same corruption: 991 → 951 frames rejected
(−40: exactly the frames the removed checks had rejected), 7 879 → 7 575 samples and 13 → 11 beats lost,
false accepts 0 → 0. Clean line unchanged: 8/8 BIT-EXACT, GATE 4 PASS.

## Hardware bring-up (week 4)

`SerialSource` (pyserial, not installed yet) is tested **before** the board is connected: on the USB-UART
adapter connect TX ↔ RX (loopback), send the golden frames of a segment from the PC and read them back
through `SerialSource` + `HostDecoder`; the run must be BIT-EXACT with the golden digest, exactly like
`--check` on a clean line. Only then connect the board.

Tests: `tests/test_uart.py` (CRC-16 and CRC-32 check values, table = bitwise, round trip, any single-bit
flip rejected, BEAT checks, SAMPLES gaps accepted + resync, resynchronisation after dropped bytes and
garbage, the measurement above),
`tests/test_stream.py` (streaming pipeline = golden, offline golden BEAT frames = streamed frames, clean
check = GATE 4, corrupted check).
