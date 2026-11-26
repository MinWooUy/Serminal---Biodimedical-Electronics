# Host ↔ Conv1D accelerator

Top FPGA: `top/ecg_accel_top.v`. Inputs `clk`, active-high `reset`, `uart_rx`;
output `uart_tx`. Connect USB-UART TX → FPGA RX, USB-UART RX ← FPGA TX, common
GND, with logic voltage compatible with the FPGA board. Board pin assignments
and timing constraints must be supplied for the actual board. This is a logic-level
UART interface, not a direct RS-232 voltage interface.

Defaults: clock 50 MHz, UART 921600 8N1, no hardware flow control. Change
`CLK_FREQ` to the actual input clock and match `BAUD_RATE` on the host.
`CLK_FREQ/BAUD_RATE >= 8`. TX holds the complete stop bit; RX synchronizes the
input, samples near bit centres, rejects false starts and bad stop bits.

This wrapper drives the existing **single-tile** Conv1D accelerator (3×3,
signed INT8 inputs/weights, INT32 results). It does not implement a complete CNN,
padding, activation, quantization, or autonomous layer scheduling. Host must
quantize ECG consistently with the model, lay out weights as `[Cout][K]`, and
schedule subsequent tiles if needed. `K=kernel_width*input_channels`;
`window_step=stride*input_channels`.

## CNN1D components

The accelerator source tree is `rtl/cnn1d/`. The UART is only the host
transport layer; the CNN1D datapath starts at `top/ecg_accel_top.v` and is
instantiated through `conv1d/conv1d_tile_top.v`.

### Top-level data path

```text
Host UART
   -> uart_rx / packet_rx
   -> host_command_controller
   -> input or weight ping-pong SRAM
   -> ifmap/weight loader and address generator
   -> ifmap/weight feeder
   -> 3 x 3 PE array and partial-sum manager
   -> INT32 tile result
   -> packet_tx / uart_tx
```

The host sends signed INT8 input samples and signed INT8 weights. The
accelerator accumulates signed INT32 results. `K` is the flattened kernel
length (`kernel_width * input_channels`), while `window_step` is the address
step between input windows (`stride * input_channels`).

### Arithmetic units: `arith/`

These modules implement the signed arithmetic used by the MAC datapath:

- `half_adder.v`, `full_adder.v`: one-bit adders.
- `CarryLookaheadAdder_*_bit.v`, `final_carry_lookahead_adder.v`: 4/8/16/32-bit carry-lookahead adders.
- `baugh_wooley_partial_product.v`: signed partial-product generation.
- `wallace_tree_reducer.v`, `wallace_tree_8bit.v`, `wallace_tree_multiplier.v`: signed multiplier reduction.

### Processing elements: `pe/`

- `ProcessingElement.v`, `pe_top.v`: MAC processing-element hierarchy.
- `MAC.v`, `Mul_Add_pipe.v`: signed multiply-accumulate and pipeline stages.
- `WeightRegister.v`, `ImageRegister.v`: hold weight and input values.
- `Accumulator.v`: stores the running partial sum.
- `shadow_register.v`, `scan_endpoint_mux.v`: state capture and scan/debug support.

### Compute core: `core/`

- `pe_array.v`: instantiates the spatial PE array; the default tile is 3 rows
   by 3 columns.
- `partial_sum_manager.v`: routes, stores and combines partial sums.
- `cnn_control_fsm.v`: sequences loading, computation and result completion.
- `cnn_compute_core.v`: top-level compute core connecting the FSM, PE array and
   partial-sum manager.

### Data feeders: `feeder/`

- `FIFO_param.v`: parameterized buffering.
- `feed_delay.v`: aligns data between feeder lanes.
- `feed_lane.v`: one lane of streamed tensor data.
- `ifmap_feeder.v`: broadcasts input activations to the PE array.
- `weight_feeder.v`: broadcasts weights to the PE array.

### Tensor memory: `mem/`

- `tensor_sram.v`: synchronous tensor storage.
- `tensor_pingpong_buffer.v`: dual-bank storage. Host writes the staging bank;
   a controlled swap makes it the active read bank without interrupting the
   current computation.

### Conv1D orchestration: `conv1d/`

- `conv1d_address_generating.v`: generates input and weight addresses for each
   window and kernel position.
- `conv1d_ifmap_loader.v`: reads input samples from the input SRAM.
- `conv1d_weight_loader.v`: reads weights from the weight SRAM.
- `conv1d_ifmap_feeder_top.v`: connects input loader to the input feeder.
- `conv1d_weight_feeder_top.v`: connects weight loader to the weight feeder.
- `conv1d_tile_top.v`: tile-level controller and integration point for SRAM,
   loaders, feeders and compute core.
- `conv1d_layer_controller.v`: higher-level sequencing for multiple Conv1D
   operations; the current UART wrapper uses the single-tile path.

### Optional post-processing: `post/`

- `conv1d_post_process.v`: optional INT32-to-INT8 bias, shift, ReLU and
   saturation stage.
- `maxpool1d2.v`: 1D max-pooling by 2.

The current `ecg_accel_top.v` returns the raw nine signed INT32 tile results.
Post-processing and a complete multi-layer CNN schedule are not automatically
enabled by the UART wrapper.

## Files

- `uart/uart_tx.v`, `uart_rx.v`: byte-level UART 8N1.
- `crc/crc16_ccitt_next.v`: combinational one-byte CRC update.
- `crc/crc_generator.v`, `crc_checker.v`: standalone streaming CRC helpers.
- `protocol/packet_rx.v`: bounded staging buffer, length/CRC validation, timeout.
- `protocol/packet_tx.v`: response framing and CRC, UART ready/valid handshake.
- `uart/uart_crc_top.v`: serial packet transport (real RX/TX ports, no loopback).
- `protocol/host_command_controller.v`: commands, bank ownership and data coverage.
- `../top/ecg_accel_top.v`: transport + controller + existing `conv1d_tile_top`.
- `host/ecg_host.py`: Python host client and a small signed Conv1D example.

## Wire format

All multibyte integers are **little-endian**. Every byte uses its own UART 8N1 frame.

```
A5 5A | CMD:u8 | SEQ:u8 | LEN:u16 | PAYLOAD:LEN bytes | CRC:u16
```

CRC-16/CCITT-FALSE: polynomial `1021`, initial `FFFF`, RefIn/RefOut false,
XorOut `0000`. Cover **CMD, SEQ, both LEN bytes, and all payload bytes**;
exclude SOF and transmitted CRC. Send CRC low byte first. Check vector ASCII
`123456789` → `29B1`. SOF bytes inside a payload are ordinary data.

Default maximum payload is 256 bytes. Responses echo SEQ; SEQ is for matching,
not deduplication. Only **one request may be outstanding**: wait for the entire
ACK/NACK/RESULT before the next request. Requests sent during execution or
response transmission are discarded. There is no RTS/CTS or request queue.

RX aborts incomplete packets after 100 ms without a byte by default
(`RX_TIMEOUT_CLKS=CLK_FREQ/10`). Framing errors and incomplete-packet timeouts
are silently dropped because their header may be unreliable. Oversized packets
and CRC failures produce NACK using the header as received (its SEQ may be corrupt).
After a malformed/oversized transfer, stop sending and allow at least the RX
timeout before resynchronizing. No automatic retries are performed by the client;
a lost response to RUN must not trigger a blind retry.

## Commands

| CMD | Request payload | Response |
| --- | --- | --- |
| `01 WRITE_IFMAP` | `base:u16`, followed by 1–254 signed INT8 ECG bytes | ACK after writes complete |
| `02 WRITE_WEIGHT` | `base:u16`, followed by 1–254 signed INT8 weights | ACK after writes complete |
| `03 CONFIG_TILE` | Legacy: `K:u16, window_step:u16, rows:u8, cols:u8, x_base:u16, w_base:u16` (10 bytes); preload: append `preload_bias:u8` = 1 (11 bytes) | ACK |
| `04 RUN_TILE` | Empty | RESULT after computation, or NACK; no preliminary ACK |
| `05 GET_STATUS` | Empty | ACK with one additional flags byte |
| `06 WRITE_BIAS` | `base:u16`, followed by 1–3 signed little-endian INT32 biases | ACK after writes complete |
| `07 CONFIG_POST` | `shift_ch0:i8, shift_ch1:i8, shift_ch2:i8, relu:u8` (`relu` must be 0 or 1) | ACK |

Reserved commands for the planned multi-layer pipeline. The numeric values are
registered in the layer controller and host for documentation, but the current
FSM does not implement them yet:

| CMD | Request payload | Planned response |
| --- | --- | --- |
| `11 CONFIG_ALL` | `layer_count:u8` + four layer configurations | ACK after all configs are stored |
| `12 LOAD_COMMIT` | checksum/flags | ACK after input, weights, and configs are validated |
| `13 RUN_ALL` | empty | RESULT after the autonomous layer pipeline |
| `14 GET_RESULT` | `result_type:u8` | RESULT with the final output |

Until the decoder and pipeline controller are connected, these commands return
NACK code 4 (unknown command) on the current bitstream.

### Layer protocol: FC layers through CONFIG flags

The layer protocol (`top/layer_cmd_controller.v`, commands `08`–`10`, `15 INFO`) runs
FC layers on the same PE array as conv: an FC layer is a conv with `Kw = 1`. CONFIG
takes an optional 12th byte `flags` (11 bytes = flags 0, so older hosts keep working):

| Bit | Flag | Effect (details in `conv1d/conv1d_layer_top.v`) |
| --- | --- | --- |
| 0 | `out_int32` | no requant: raw INT32 of row 0, little-endian, 4 bytes per channel at address `4*c` of the output bank (`out_len = 1`) |
| 1 | `argmax` | with bit 0: after the layer, the index of the largest channel (ties → smallest index) at address `4*cout` |
| 2 | `row_combine` | 3 PE rows of one channel merge into one output: `h = clamp(rsr(rsr((P1<<7)+P0, SG) + P2 + bias, SO), 0, 127)` (`out_len = 3`, ≥ 3 PE rows) |
| 3 | `gap_in` | with bit 2: before running, pack `gap_mem` into the write bank as row 0 = `gap & 127`, row 1 = `gap >> 7`, row 2 = zeros up to the GAP channel count (the host writes the rest of row 2, e.g. the 4 RR bytes), then swap |

CONFIG may also carry bytes 13–16 `w_base:u16, b_base:u16` (16-byte CONFIG): the layer reads its
weights from the READ weight bank at `w_base + c*K + k` and its bias from `bias_mem[b_base + c]`
(bias RAM: `BIAS_DEPTH = 256` entries, `WRITE_BIAS` accepts addresses up to it). With this the whole
CNN Lite stays on chip: bank A holds conv0, conv1, conv2, fc1, fc2 (11 120 B), bank B holds conv3
(12 288 B), all 212 biases sit in the bias RAM; the host loads them once and per beat only sends the
input, the shifts, CONFIG/RUN (and a weight SWAP around conv3). INFO byte 3 bit 4 (`0x10`) reports it.

`WRITE_SHIFT` addresses `0..COUT_MAX-1` hold the requant shift (SO); addresses
`COUT_MAX..2*COUT_MAX-1` hold the pre-shift SG used by `row_combine`. INFO version 2
reports the supported features in byte 3 (`0x3F`; bit 5 `0x20` = layers with pool use the largest
even row count `ARRAY_ROWS & ~1`, e.g. 6 rows on 7×7, so the host plans tiles with that many rows). CNN Lite uses FC1 = `row_combine | gap_in`
(`out_len 3, cout 32, K 68`) and FC2 = `out_int32 | argmax` (`out_len 1, cout 4, K 32`);
`host/ecg_layer_host.py` runs conv0 → fc2 and reads the 4 logits + label with `READ_FMAP 0, 17`.

Response commands:

- `80 ACK`: payload `[original_CMD, 00]`; GET_STATUS adds flags at byte 2:
  bit 0 = configuration valid, bit 1 = input staging bank dirty,
 bit 2 = weight staging bank dirty; bit 3 = bias preload enabled; bit 4 = post configured; bits 7:5 reserved zero.
- `81 NACK`: payload `[original_CMD, error]`:
  `01` CRC/oversized packet, `02` invalid length/config/address,
  `03` unconfigured/not runnable/missing tensor data, `04` unknown command.
- `82 RESULT`: with post disabled, exactly 36 bytes: nine signed little-endian
  INT32 values in physical row-major order `(row*3+col)*4`. With the top-level
  `POST_ENABLE=1`, exactly 45 bytes: those 36 raw bytes followed by nine signed
  INT8 post values in the same row-major order. Inactive rows/columns are zero.

For default parameters, `1 <= K <= 16`, `1 <= rows,cols <= 3`,
`window_step > 0`. All input/weight addresses must fit the 1024-byte SRAM banks.
CONFIG validates the last required address using extended arithmetic.
CRC-invalid packets never write SRAM, change configuration, swap banks or start
computation. The controller validates all required addresses before RUN.

## Bank lifecycle

1. After reset neither bank contains valid host data.
2. WRITE commands always fill the **write/staging** bank. Multiple chunks are
   allowed and may use nonzero addresses. A per-address validity bitmap records
   which bytes have actually been written.
3. RUN checks each needed input/weight address. A dirty staging bank must contain
   every needed byte; an untouched staging bank uses the previous active bank.
4. RUN swaps only dirty banks, clears their staging validity maps, then pulses
   accelerator start on a later clock. No swaps occur on a rejected RUN.
5. The controller waits for tile completion, snapshots all results, and sends
   RESULT. Repeated RUN with no writes reuses both active tensors. New ECG alone
   swaps only the input bank; weights remain available.

A partial update after a previous RUN is **not** an in-place patch of the active
bank: supply all bytes needed by the next tile to that staging bank. Missing data
causes NACK. Reset discards configuration and all validity maps; reload afterward.
The controller serializes load and compute. It intentionally does not overlap
host requests with execution in this first implementation.

## Host example

Install `pyserial` in the host Python environment, then from `rtl/cnn1d`:

```sh
python3 comm/host/ecg_host.py /dev/ttyUSB0 --baud 921600
# Windows: python comm/host/ecg_host.py COM3
```

The example loads X=`[1,-2,3,4,-1]`, three filters of length 3, and checks:

```
13   2  -5
 5   5   8
-1   6   5
```

For real ECG, use `Accelerator.write_tensor(1, samples)`,
`write_tensor(2, weights)`, `configure(...)`, then `run()`. The client checks
INT8 ranges, chunks writes, verifies response CRC/sequence and raises on NACK.
Host helper limits assume the default 1024-byte banks and K<=16.
The FPGA wrapper supports `MEM_DEPTH` up to 65536, `FEEDER_DEPTH` from 1 to 65535,
and `MAX_PAYLOAD` from 36 to 256; use default settings unless adapting the host.
Validity maps use registers and dynamic indexing; synthesis area/timing must be
checked on the target FPGA. No pin constraints or board programming are included.

## Verification

With Icarus Verilog and Python 3 installed:

```sh
sim/run_comm_tests.sh
```

The script runs CRC known-vector/error/timing tests; all 256 UART byte values,
back-to-back traffic, asynchronous baud mismatch and framing recovery; physical
UART end-to-end through SRAM and Conv1D, negative inputs, repeated runs,
input-only bank replacement, partial tile, invalid CRC/config/address, incomplete
staging data and packet timeout. Host framing/error handling is tested separately.
UART unit tests use the default 50 MHz/921600 divider. The focused timing test
checks the resulting DIV=54 (1.08 us per bit). End-to-end tests use a
smaller divider for speed; they still serialize every byte over RX/TX pins.

ModelSim can use the updated `filelist.f` and `tb/tb_uart_crc_top.v`, top module
`tb_uart_crc_top`. Testbenches use `$fatal` and explicit timeouts. The old UART
interfaces (24-bit custom frame and internal loopback) have been replaced.

## Evaluation scope and multi-layer data-flow plan

The software model is trained and frozen in Python. The FPGA is an inference
device only: it receives the frozen quantized parameters and input vectors,
computes the forward pass, and returns results for comparison with the Python
integer/golden reference. The two sides must use the same model parameters,
quantization, preprocessing, padding, record/segment, beat, and tensor layout.

The current evaluation scopes are intentionally different:

- Python detector/model evaluation covers all 22 DS1 records. DS2 is a separate
   locked evaluation and requires explicit user authorization.
- The exported FPGA golden data defaults to 8 DS1 records, one selected
   120-second segment per record, about 1,306 beats in total.
- `host/ecg_layer_host.py` defaults to one beat (`rec101_93600`, beat 0) and
   runs layers 0..3. It is a bit-exact smoke/diagnostic runner, not a complete
   record evaluation.

For a fair FPGA result, Python and FPGA must evaluate the same golden segment
and beat set. Layer-level checks should compare `z_q`, every Conv/Pool output,
GAP, logits, and prediction; the final report should aggregate results per
record and over the same selected set.

The intended multi-layer data flow is a feature-map ping-pong pipeline:

```text
Conv0: read Bank A -> write Bank B -> swap
Conv1: read Bank B -> write Bank A -> swap
Conv2: read Bank A -> write Bank B -> swap
Conv3: read Bank B -> GAP/logits
```

The configuration for all layers must be copied into dedicated configuration
registers or a separate configuration RAM before execution, for example
`cfg[0]` through `cfg[3]`. It must not be stored only in a bank that will later
be overwritten by a feature-map output. Configuration includes layer geometry,
pool/GAP/padding controls, bias, and shift. The feature-map banks should hold
only data; weight storage should be separate.

For the full CNN-Lite layer path (the `layer_core`/`de10_lite_top` configuration,
not the default single-tile helper described above), the largest feature-map is
about 3,008 signed INT8 bytes. Therefore two `IFMAP_MEM_DEPTH=4096` banks are
sufficient for the alternating feature maps:

| item | size |
| --- | ---: |
| Conv0 input with padding | 366 B |
| Conv0 output after pool/padding | 2,944 B |
| Conv1 output after pool/padding | 2,944 B |
| Conv2 output after pool/padding | 3,008 B |
| Conv3 output before GAP | 2,880 B |

The four convolution weight tensors total 21,104 B (112 B, 2,560 B,
6,144 B, and 12,288 B). The current 16,384-byte weight memory can hold each
layer separately, but not all four tensors in one contiguous region. Without a
larger weight memory, load each layer's weights once when changing layers and
reuse them for all beats; do not resend them for every beat. To load the whole
model once and retain all weights, use a separate weight memory of at least
21,104 B (32 KiB is a practical rounded size), or a dedicated ROM/BRAM layout.

The current frame remains:

```text
A5 5A | CMD:u8 | SEQ:u8 | LEN:u16 LE | PAYLOAD | CRC16:u16 LE
```

With `MAX_PAYLOAD=256`, one tensor write carries at most 254 data bytes after
the 2-byte base address. A complete model transfer therefore consists of
multiple valid frames, not one 20-KiB frame. The documented and tested UART
configuration is 921600 baud. The host and FPGA must use the same rate; with
the 50 MHz board clock the integer divider is 54, giving an actual baud rate
of about 925926 baud (0.47% fast). Regardless of baud rate, the
current protocol is stop-and-wait: one request is sent and its ACK/RESULT is
consumed before the next request. The current RTL has no request FIFO, so
sending frames blindly back-to-back loses data.

If burst loading is implemented later, retain the existing CMD meanings and
CRC on every frame, write each valid block directly to staging memory, and
return an ACK only at a commit boundary such as `SWAP`. A block identifier or
address range must still be tracked so a missing/corrupt block is detected;
removing per-block ACKs must not remove per-frame CRC validation. This requires
changes to the UART/controller FSM and is not implemented by the current
single-tile wrapper.

The desired end-to-end sequence is therefore:

```text
load all layer configuration -> load input/data -> load weights as supported
-> commit/swap staging data -> RUN_ALL in the autonomous controller
-> Conv0 -> Pool -> Conv1 -> Pool -> Conv2 -> Pool -> Conv3 -> GAP/FC
-> return final result to the PC -> compare with Python
```

The existing RTL implements the single-tile command path documented above. A
true `RUN_ALL` flow additionally needs per-layer configuration storage, an
autonomous layer FSM, protected configuration memory, and a weight-memory
policy. These are architectural extensions, not assumptions about the current
bitstream.
