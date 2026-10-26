# Giao thức PC ↔ FPGA, phiên bản 2 (chỉ CNN)

> **Nguồn gốc là đặc tả giao tiếp dùng chung với nhóm FPGA.** File này phải khớp với đặc tả đó (CLAUDE.md, quy tắc 8).
> Mọi thay đổi khung, mã TYPE, độ dài, ý nghĩa trường hay bảng mã NACK phải báo nhóm FPGA trước, tăng `PROTO_VER`
> và ghi vào `NHATKY.md`.
>
> Bản cài đặt tham chiếu: `python/link/proto_v2.py` (chỉ dùng thư viện chuẩn), test: `tests/test_proto_v2.py`.
> Giao thức v1 (`docs/uart_protocol.md`, thiết bị mô phỏng chạy cả pipeline) là **chuyện khác**: không bao giờ trộn v1 và
> v2 trên cùng một đường truyền (hai bên dùng chung mã 0x01 với nghĩa khác nhau).

## 1. Vai trò

- **PC** làm toàn bộ tiền xử lý số nguyên (lọc, Pan-Tompkins, R, cửa sổ, z-score, RR) và gửi cho FPGA từng nhịp đã sẵn sàng:
  `z_q` (360 × int8, thang 2^-4) và `rr_q` (4 × int8, thang 2^-6, 2^-6, 2^-6, 2^-5).
- **FPGA** chạy CNN int8 (Lite đóng băng) và trả 4 logit int32 (thang chung 2^-11) + nhãn `pred` = argmax
  (hoà → chỉ số nhỏ nhất; 0 N, 1 S, 2 V, 3 F) + số chu kỳ đã dùng.
- Kết quả phải **trùng từng bit** với golden reference (`python/quant/int8.py:int_forward_q`).

## 2. Lớp vật lý

UART **115 200 baud, 8N1**. Một khung BEAT_IN (375 byte) mất ≈ 32.6 ms trên dây; RESULT (32 byte) ≈ 2.8 ms.

## 3. Khung

| offset | kích thước | trường |
|---|---|---|
| 0 | 1 | `0xA5` |
| 1 | 1 | `0x5A` |
| 2 | 1 | `TYPE` |
| 3 | 2 | `LEN` = độ dài payload, **u16 little-endian** |
| 5 | `LEN` | payload |
| 5 + `LEN` | 2 | `CRC`, **little-endian** |

- **CRC-16/CCITT-FALSE**: đa thức 0x1021, khởi tạo 0xFFFF, MSB trước (không đảo bit), xorout 0; tính trên
  **TYPE + LEN (2 byte) + PAYLOAD** (không gồm sync). Giá trị kiểm: CRC("123456789") = **0x29B1**.
- Mọi trường nhiều byte là little-endian; số có dấu dạng bù 2.
- Ví dụ: khung PING = **`A5 5A 01 00 00 AC FB`** (CRC của `01 00 00` = 0xFBAC).

## 4. Các gói

| TYPE | Tên | Chiều | LEN | Payload |
|---|---|---|---|---|
| 0x01 | PING | PC → FPGA | 0 | — |
| 0x02 | PONG | FPGA → PC | 10 | `proto_ver u8` (= 2), `mode u8` (= 1: chỉ CNN), `build_id u32`, `weights_crc32 u32` |
| 0x10 | BEAT_IN | PC → FPGA | 368 | `beat_id u32`, `z_q i8 × 360`, `rr_q i8 × 4` |
| 0x11 | RESULT | FPGA → PC | 25 | `beat_id u32`, `logit i32 × 4`, `pred u8`, `cycles u32` |
| 0x1F | NACK | FPGA → PC | 4 | `reason u8`, `last_type u8`, `err_count u16` |

- `build_id`: nhóm FPGA tự đặt (gợi ý `0xYYMMDDNN`); PC chỉ in ra, không kiểm.
- `weights_crc32`: CRC-32/IEEE trên **byte thô** của mọi mảng tham số, **đúng thứ tự file trong `mem/`**, little-endian.
  FPGA chỉ trả nguyên hằng số `WEIGHTS_CRC32` trong `ecg_params.vh` (không phải tự tính). PC so với giá trị tính từ
  `lite_int8.npz` đóng băng; khác → từ chối chạy.
- `RESULT.beat_id` = `beat_id` của BEAT_IN tương ứng. `cycles`: số chu kỳ FPGA dùng cho nhịp đó (chỉ để báo cáo, không so golden).

### NACK

| reason | Ý nghĩa |
|---|---|
| 1 | CRC sai — **vẫn gửi NACK**; `last_type` khi đó chỉ để tham khảo (byte TYPE có thể đã hỏng), PC không dựa vào |
| 2 | LEN không đúng với TYPE |
| 3 | TYPE lạ |
| 4 | đang bận |

`err_count` = tổng số khung lỗi FPGA đã gặp kể từ lúc reset.

## 5. Bên nhận (giải khung)

1. Tìm `A5 5A`.
2. Đọc TYPE và LEN. TYPE không thuộc các gói bên này nhận → lỗi reason 3. LEN khác độ dài cố định của TYPE → lỗi reason 2.
   Hai kiểm tra này làm **trước** khi chờ payload, nên một LEN bị hỏng (ví dụ 0xFFFF) không làm bộ nhận đứng chờ.
3. Đủ byte → kiểm CRC; sai → lỗi reason 1.
4. Mọi lỗi: **bỏ đúng 1 byte** rồi quay lại bước 1 (tự đồng bộ). Phía FPGA gửi NACK cho mỗi lỗi; phía PC chỉ đếm.

## 6. Stop-and-wait (PC)

- PING → chờ PONG **500 ms**. BEAT_IN → chờ RESULT **1 s**.
- Hết giờ hoặc nhận NACK → gửi lại **ngay**, giữ nguyên `beat_id`; **tối đa 3 lần gửi lại** (NACK cũng tính vào giới hạn này).
- RESULT có `beat_id` khác → bỏ qua, chờ tiếp (không tính là lỗi). FPGA không cần chống trùng: nhận lại cùng nhịp thì tính lại.

## 7. Diễn giải của phía PC — cần nhóm FPGA xác nhận

Các điểm sau đặc tả chưa nói rõ; `python/link/` đang cài theo cách dưới đây:

1. "Gửi lại tối đa 3 lần" = **3 lần gửi lại sau lần gửi đầu** → tối đa 4 lần gửi cho mỗi gói.
2. FPGA nhận một khung đúng CRC nhưng TYPE là gói **chiều ngược lại** (PONG, RESULT, NACK) → coi là **TYPE lạ** (reason 3).
3. Lệnh `golden`: `beat_id` = chỉ số toàn cục của nhịp golden, 0…1 305, theo thứ tự đoạn trong `export/golden_manifest.json`
   rồi thứ tự nhịp trong đoạn (trùng với `golden/result_expected.hex`).

## 8. File cho nhóm FPGA (`scripts/export_rtl.py` → `export/rtl/`, không commit, tái tạo được)

- `ecg_params.vh`: `localparam` (include **bên trong** module): kích thước từng lớp, có/không MaxPool, mã TYPE và LEN,
  bảng NACK, `LOGIT_FRAC = 11`, độ sâu / độ rộng / địa chỉ byte của từng mảng, `WEIGHTS_BYTES`, `WEIGHTS_CRC32`,
  `PROTO_VER = 2`, `MODE = 1`; đầu file chép công thức tính từng lớp (dịch có làm tròn, ReLU + kẹp, FC1 hai khối).
- `mem/NN_<tên>.hex` (+ `.mif` cho Quartus): **mỗi mảng một bộ nhớ**, `$readmemh`, **mỗi dòng một giá trị**, bù 2,
  int8 = 2 chữ số hex, int32 = 8 chữ số. Số thứ tự `NN` (00…19) **là thứ tự hợp đồng** dùng cho `WEIGHTS_CRC32`:
  `00 conv0_w · 01 conv0_b · 02 conv0_shift · 03–05 conv1 · 06–08 conv2 · 09–11 conv3 · 12 fc1_wg · 13 fc1_wr ·
  14 fc1_b · 15 fc1_shift_g · 16 fc1_shift_r · 17 fc1_shift_out · 18 fc2_w · 19 fc2_b`.
  Thứ tự chỉ số: trọng số conv co → ci → k; trọng số FC out → in; `*_shift` theo kênh / hàng.
  Tổng 24 528 byte; `WEIGHTS_CRC32 = 0x067B9985` (mô hình Lite đóng băng).
  Nhóm FPGA có thể tự ghép nhiều giá trị vào một từ nhớ nếu muốn; file vẫn giữ một giá trị mỗi dòng.
- `golden/`:
  - `beats_sel.hex`: `beat_id` (int32) của các nhịp có đủ mọi lớp (mặc định 20 nhịp mỗi lớp thật N/S/V/F = 80;
    `--all-beats`: cả 1 306, khoảng 111 MB);
  - `zq.hex`, `rrq.hex`, `conv0…conv3.hex`, `pool0…pool2.hex` (int8), `gap.hex` (int32), `fc1.hex` (int8):
    các nhịp đã chọn, xếp theo nhịp → kênh c → vị trí t (`conv*` là đầu ra sau dịch / ReLU / kẹp, **trước** MaxPool);
  - `logits.hex` (int32 × 4) và `pred.hex` (u8) cho **cả 1 306 nhịp**;
  - `result_expected.hex`: mỗi nhịp 6 từ 32 bit `beat_id, logit0, logit1, logit2, logit3, pred` (không có `cycles`);
    testbench giải RESULT rồi so các trường này;
  - `uart_in.bin`: 1 306 khung BEAT_IN nối tiếp nhau, `beat_id` 0…1 305.
- `manifest.json`: hình dạng, ý nghĩa, SHA-256 của từng file.
