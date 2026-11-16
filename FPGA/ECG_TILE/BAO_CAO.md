# Báo cáo bàn giao — CNN ECG trên bộ tăng tốc theo ô

Gói `ECG_TILE_20261116_684553e` · ngày 2026-11-16 · commit dự án `684553e423e7` · RTL đối chiếu Serminal `dev` `4ad6c8b` ·
mô hình `2026-10-15_581b4c74_final_A5_svdb_n25_lite_e3` (SHA-256 của `lite_int8.npz` bắt đầu `4940bef2dc096a2c`). Mọi số dưới đây được tính / đo khi dựng chính gói này.

## 1. Đã làm gì
1. **Đặc tả giao thức ô** (`docs/proto_tile.md`) viết lại từ RTL `4ad6c8b`: khung có SEQ, 7 lệnh, ACK / NACK / RESULT, vòng đời bank,
   công thức một ô; chỗ README của RTL khác RTL thì theo RTL.
2. **Lập lịch theo ô**: mô hình Lite đóng băng chạy trên ô 7×7 — conv0–3, FC1 phần RR, FC2 trên FPGA; MaxPool, GAP, ghép FC1, argmax trên PC.
3. **Bộ tăng tốc giả lập mức byte** theo `packet_rx.v` + `host_command_controller.v` (NACK 1–4, timeout 100 ms, bỏ byte khi bận).
4. **Host** dừng-chờ, không tự gửi lại, chẩn đoán lỗi bằng tiếng Việt, ghi mọi byte; cùng một mã chạy với board (`--port`) hoặc giả lập (`--sim`).
5. **Vector cho testbench RTL**: `golden/tile_trace/` — toàn bộ byte gửi / nhận của 4 nhịp N / S / V / F.

## 2. Kết quả kiểm khi dựng gói này
- Bộ test dự án: `364 passed, 1 warning in 397.31s (0:06:37)`.
- Host của chính gói (chạy một mình, numpy + pyserial, trên bộ giả lập):
  - `smoke` → ô 3x3: nhận [[-2, 0, 4], [-6, -1, 11], [4, 10, 2]], mong đợi [[-2, 0, 4], [-6, -1, 11], [4, 10, 2]] -> ĐẠT
  - `layers` → 80/80 nhịp trùng từng bit, nhãn đúng 80/80; 0 nhịp lỗi đường truyền; 134480 giao dịch, 20765840 B, 1823.3 s (22.79 s / nhịp)
  - `golden --all` → 1306/1306 nhịp trùng từng bit, nhãn đúng 1306/1306; 0 nhịp lỗi đường truyền; 2195386 giao dịch, 339002338 B, 29764.7 s (22.79 s / nhịp)
- **Chưa chạy trên board thật.** Nhóm FPGA đã tự kiểm conv0 trên board: 5 760 / 5 760 giá trị trùng golden.

## 3. Chi phí đường truyền mỗi nhịp (115 200 baud, 8N1, chỉ thời gian trên dây)
Mỗi nhịp: **513 ô, 1 681 giao dịch, 259 573 byte ≈ 22.5 s**; 1 306 nhịp ≥ 8.2 giờ.
Riêng các khung RESULT: 129 789 B ≈ 11.3 s (mỗi ô trả 196 B raw + 49 B post).

| Lớp | Kênh / lần nạp trọng số | Vị trí / lần nạp ifmap | Ô | Giao dịch | Byte | Giây |
|---|---|---|---|---|---|---|
| conv0 | 16 | 360 | 156 | 321 | 45 980 | 3.99 |
| conv1 | 7 | 56 | 130 | 366 | 53 804 | 4.67 |
| conv2 | 7 | 28 | 130 | 466 | 70 796 | 6.15 |
| conv3 | 5 | 14 | 91 | 511 | 86 850 | 7.54 |
| fc1_rr | 32 | 1 | 5 | 12 | 1 617 | 0.14 |
| fc2 | 4 | 1 | 1 | 5 | 526 | 0.05 |

Cùng chuỗi lệnh nếu thay đổi phía FPGA: RESULT chỉ trả post → 13.8 s; 460 800 baud → 5.6 s;
921 600 baud → 2.8 s; 921 600 baud + chỉ post → 1.7 s. Chưa tính thời gian tính ô trên FPGA và độ trễ
USB-UART mỗi lượt (1 681 lượt × 1…16 ms = thêm 1.7…27 s).

## 4. Hạn chế đã biết
- **Tốc độ**: xem mục 3; chạy trên board nên dùng tập nhỏ (README bước 3–6).
- **Không gửi lại**: một khung lỗi làm hỏng cả nhịp (3 362 khung hai chiều / nhịp). Tỉ lệ lỗi khung 1e-5 / 1e-4 / 1e-3 → nhịp qua
  96.7 % / 71.4 % / 3.5 %. Nhịp lỗi được báo, không bao giờ cho ra số sai (đã thử bằng tiêm lỗi).
- **MaxPool, GAP, ghép FC1 ở PC**: `maxpool1d2.v` chưa nối vào top; đầu vào FC1 phần GAP ≈ 14 bit, ô chỉ nhận int8.
- **Thời gian tính một ô** chưa đo: giả lập giả định 0,5 ms / ô.

## 5. Bố cục dữ liệu cần nhớ
ifmap `X[t][ci]`, `step = Cin`, `K = k·Cin`; trọng số mỗi kênh ra theo `[k][ci]` (khác `co→ci→k` của gói v2 cũ); đệm "same" là các số 0 host
thêm vào. conv3: K = 192 nên tối đa 5 cột mỗi ô (bank trọng số 1 024 B). Shift của mô hình đều trong 7…10.
