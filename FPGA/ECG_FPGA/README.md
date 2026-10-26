# Gói bàn giao FPGA — CNN phân loại nhịp ECG (int8, giao thức v2)

Gói: `ECG_FPGA_20261026_e64a4ab` · commit `e64a4aba7214` · WEIGHTS_CRC32 = **0x067B9985** · chi tiết trong `VERSION`.

## 1. Cài đặt (một lần)
- Python 3.10 trở lên (python.org; khi cài, tick "Add python.exe to PATH").
- Mở PowerShell trong thư mục gói: `pip install -r requirements.txt` (numpy, pyserial).
- Driver của module USB–UART (CH340 / CP210x / FTDI). Module **đặt 3.3 V**, không dùng 5 V với chân FPGA.
- Cổng COM của module: Device Manager → Ports (COM & LPT). Dưới đây dùng `COM5`, thay bằng cổng của bạn.

## 2. Chạy theo thứ tự — chỉ sang bước sau khi bước trước ĐẠT
| Bước | Chuẩn bị | Lệnh | Kết quả mong đợi |
|---|---|---|---|
| 0 | không cần board | `python ref/ref_cnn.py --rtl .` | `BIT-EXACT with golden` |
| 1. loopback | nối tắt TX–RX trên module, **chưa** nối FPGA | `python host/fpga_host.py loopback --port COM5` | `LOOPBACK ĐẠT` |
| 2. ping | tháo dây nối tắt; TX module → RX FPGA, RX module → TX FPGA, GND chung; nạp bitstream | `python host/fpga_host.py ping --port COM5` | `PING ĐẠT` (weights_crc32 = 0x067B9985) |
| 3. golden | như bước 2 | `python host/fpga_host.py golden --port COM5` | `1306/1306 logit trùng từng bit ... ĐẠT` |

- Khung PING đúng là `A5 5A 01 00 00 AC FB`; UART 115200 8N1.
- Mỗi lần chạy tạo `ket_qua/<ngày_giờ>_<lệnh>/` gồm `log.txt`, `result.json`, `raw_bytes.txt` (mọi byte gửi/nhận
  kèm thời gian). `--raw-log <file>` để ghi byte thô ra file khác; `--limit 20` để thử golden với 20 nhịp đầu.
- Ping thất bại → chương trình in chẩn đoán (không có byte nào về / có byte nhưng không ra khung / NACK / sai trọng số).

## 3. Khi có lỗi, gửi lại cho phía PC
1. Nguyên thư mục `ket_qua/<ngày_giờ>_<lệnh>/` của lần chạy lỗi (nén zip): `log.txt`, `result.json`, `raw_bytes.txt`.
2. Ảnh chụp cửa sổ lệnh; ảnh Device Manager (cổng COM); ảnh/video board và dây nối (TX, RX, GND, công tắc 3.3 V).
3. `build_id` in trong PONG (nếu có), file `VERSION` của gói, và commit/bitstream phía FPGA đang dùng.

## 4. Tài liệu trong gói
- **Đặc tả giao tiếp dùng chung (bản gốc): (chưa điền — chạy lại make_fpga_handoff.py --spec-url <link>)**
- `PROTO_V2.md`: đặc tả phía PC (phải khớp bản gốc) — khung, gói, NACK, stop-and-wait, định dạng file.
- `params/ecg_params.vh`: kích thước, số mũ dịch, địa chỉ, WEIGHTS_CRC32, công thức từng lớp.
- `mem/NN_<tên>.hex` / `.mif`: tham số (thứ tự 00…19 là thứ tự tính WEIGHTS_CRC32). `golden/`: kết quả chuẩn từng lớp
  (80 nhịp, 20 mỗi lớp N/S/V/F), `logits.hex`, `pred.hex`, `result_expected.hex`, `uart_in.bin` (1306 nhịp).
- `ref/ref_cnn.py`: mô hình tham chiếu chỉ dùng numpy — đọc đúng các file trên.
- Kiểm toàn vẹn: `SHA256SUMS` (PowerShell: `Get-FileHash -Algorithm SHA256 <file>`).
