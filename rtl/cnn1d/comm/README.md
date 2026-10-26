# Host ↔ Conv1D accelerator

Top FPGA: `top/ecg_accel_top.v`. Inputs `clk`, active-high `reset`, `uart_rx`;
output `uart_tx`. Connect USB-UART TX → FPGA RX, USB-UART RX ← FPGA TX, common
GND, with logic voltage compatible with the FPGA board. Board pin assignments
and timing constraints must be supplied for the actual board. This is a logic-level
UART interface, not a direct RS-232 voltage interface.

Defaults: clock 50 MHz, UART 115200 8N1, no hardware flow control. Change
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
python3 comm/host/ecg_host.py /dev/ttyUSB0 --baud 115200
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
UART unit tests use the default 50 MHz/115200 divider. End-to-end tests use a
smaller divider for speed; they still serialize every byte over RX/TX pins.

ModelSim can use the updated `filelist.f` and `tb/tb_uart_crc_top.v`, top module
`tb_uart_crc_top`. Testbenches use `$fatal` and explicit timeouts. The old UART
interfaces (24-bit custom frame and internal loopback) have been replaced.
