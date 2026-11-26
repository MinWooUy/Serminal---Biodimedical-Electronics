# Gói bàn giao — CNN ECG trên bộ tăng tốc theo ô (giao thức ô)

Gói `ECG_TILE_20261116_684553e` · commit `684553e423e7` · RTL đối chiếu: Serminal `dev` `4ad6c8b` · chi tiết trong `VERSION`.
Đặc tả: `docs/proto_tile.md`. Báo cáo kết quả đã chạy (số thật của chính gói này): `BAO_CAO.md`.

PC điều phối mô hình Lite đóng băng theo từng ô: conv0–3, phần RR của FC1 và FC2 chạy trên FPGA (`RUN_TILE`); PC làm MaxPool, GAP,
ghép FC1, argmax. Mỗi nhịp: 513 ô, 1 681 giao dịch, ≈ 22,5 s ở 115 200 baud.

## 1. Cài đặt (một lần)
- Python 3.10 trở lên; trong thư mục gói: `pip install -r requirements.txt` (numpy, pyserial).
- Module USB–UART 3.3 V (CH340 / CP210x / FTDI). TX module → `uart_rx` FPGA, RX module ← `uart_tx` FPGA, **GND chung**.
- Bitstream: `de10_lite_top` (50 MHz, 115 200, `POST_ENABLE = 1`, ô 7×7, `MEM_DEPTH = 1024`). Reset = KEY0.

## 2. Chạy theo thứ tự — chỉ sang bước sau khi bước trước ĐẠT (thay `COM5` bằng cổng của bạn)
| Bước | Lệnh | Mong đợi | Thời gian |
|---|---|---|---|
| 0. không cần board | `python host/handoff_tile.py smoke --sim` | `ĐẠT` | tức thì |
| 1. trạng thái | `python host/handoff_tile.py status --port COM5` | `flags 0` ngay sau reset | < 1 s |
| 2. ô 3×3 | `python host/handoff_tile.py smoke --port COM5` | `[[-2, 0, 4], [-6, -1, 11], [4, 10, 2]] … ĐẠT` | < 1 s |
| 3. conv0 | `python host/handoff_tile.py layer --upto conv0 --port COM5` | `1/1 nhịp trùng từng bit … ĐẠT` | ≈ 4 s |
| 4. tới conv3 | `python host/handoff_tile.py layer --upto conv3 --port COM5` | `ĐẠT` | ≈ 22 s |
| 5. 4 lớp đầu → logit | `python host/handoff_tile.py layers --beats 0:4 --port COM5` | `4/4 … nhãn đúng 4/4 … ĐẠT` | ≈ 1,5 phút |
| 6. 80 nhịp mọi lớp | `python host/handoff_tile.py layers --port COM5` | `80/80 … ĐẠT` | ≈ 30 phút |
| 7. logit 1 306 nhịp | `python host/handoff_tile.py golden --all --port COM5` | `1306/1306 … ĐẠT` | ≈ 8,3 giờ |

Mỗi lần chạy tạo `ket_qua/<ngày_giờ>_<lệnh>/`: `log.txt`, `result.json`, `raw_bytes.txt` (mọi byte gửi / nhận, có thời gian) — gửi
cả thư mục này khi cần phân tích lỗi. Thêm `--sim` thay `--port` để chạy cùng lệnh trên bộ giả lập (không cần board).

## 3. Khi KHÔNG ĐẠT
- `không có ACK/RESULT … sau 1.0 s`: dây TX/RX đảo, thiếu GND, sai cổng COM, chưa nạp bitstream, chưa nhả KEY0.
- `NACK 1`: khung hỏng trên dây (nhiễu, sai baud). `NACK 2`: độ dài / cấu hình không hợp lệ với bitstream (ô không phải 7×7?
  CONFIG_POST phải 8 byte). `NACK 3`: RUN khi thiếu cấu hình / bias / dữ liệu (FPGA vừa reset?). `NACK 4`: bitstream không biết lệnh.
- `SAI ở ['conv0', …]`: số khác golden — xem lớp đầu tiên bị sai; `golden/tile_trace/*_tiles.txt` có từng ô mong đợi.
- Host **không bao giờ tự gửi lại** (theo quy định của nhóm FPGA). Một khung lỗi làm hỏng cả nhịp; nhịp đó được báo, nhịp sau chạy lại từ đầu.

## 4. Vector cho testbench RTL (`golden/tile_trace/`)
Với mỗi nhịp `<lớp>_beat<id>`: `_tx.bin` là toàn bộ byte host gửi (đúng thứ tự, SEQ 0, 1, …, quay vòng 256), `_rx.bin` là toàn bộ byte
FPGA phải trả (ACK / RESULT), `_tiles.txt` một dòng cho mỗi `RUN_TILE` (cấu hình ô, 49 raw, 49 post). Phát `_tx.bin` vào `uart_rx`,
**đợi trọn từng trả lời** trước khi phát khung sau (dừng-chờ), so byte `uart_tx` với `_rx.bin`.

## 5. Tệp tham số (`params/`)
`model.npz`: mô hình Lite int8 (SHA-256 `lite_int8.npz` trong `VERSION`). `tile_layout/`: đúng các byte host ghi — trọng số conv
theo kênh ra, mỗi kênh `[k][ci]` (khác thứ tự `co→ci→k` của gói v2 cũ), bias int32, shift int8 (= k + a_in − a_out), dạng `$readmemh`.
