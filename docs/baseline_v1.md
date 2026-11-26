# Mốc chốt v1 (24/11/2026): số liệu dùng cho tiểu luận

Tài liệu này **đóng băng** trạng thái hiện tại của thiết kế và số đo để viết tiểu luận. Mọi con số trong báo cáo, slide và
`report/DINHVI.md` phải truy được về bảng ở đây. Các cải tiến phần cứng tiếp theo nằm ở [`plan_cai_tien_v2.md`](plan_cai_tien_v2.md)
và **không** làm thay đổi mốc này.

## 1. Định danh

| Mục | Giá trị |
|---|---|
| Nhánh, commit gốc | `dev`, `de02453` (24/11/2026). Chưa commit tại thời điểm viết: sửa `layer_plan` của hai host, test mới, `scripts/tile_cycle_model.py`, các tài liệu này |
| Bitstream | `rtl/cnn1d/output_files/cnn1d.sof`, 21/11/2026 14:52, 3 216 551 B, SHA-256 `6f4c3b6a511bf0ee00054cbae4315907b17498b626109713a013c55530c21168` |
| Công cụ | Quartus Prime Lite 18.1.0 Build 625, chip MAX 10 `10M50DAF484C7G`, mô hình thời gian và công suất "Final" |
| Cấu hình RTL | mảng PE 7×7, 50 MHz, UART 1,5 Mbaud 8N1, INFO `features` = `0x3F` (có `w_base`/`b_base` và pool dùng hàng chẵn), IFMAP 4 096 B, trọng số 16 384 B mỗi bank (2 bank), FEEDER 1 024, COUT_MAX 64, MAX_PAYLOAD 128 |
| Mô hình | CNN Lite int8 (conv k = 7, 5, 3, 3; GAP; FC 68→32→4), 23 408 B trọng số, 212 bias; `export/final_models_manifest.json`, `weights.h` SHA-256 `1c84f26b…56d0032e` |
| Golden | `export/golden`, 8 đoạn, **1 306 nhịp** (`export/golden_manifest.json`) |
| Chuỗi kiểm chứng | float → int8 Python (golden) → mô phỏng RTL → board qua UART |

## 2. Số đo đã chốt

| # | Đại lượng | Giá trị | Nguồn / cách tái hiện |
|---|---|---|---|
| 1 | Bit-exact trên board | **1 306/1 306** nhịp, digest từng đoạn trùng golden, 0 lỗi | `results/hardware_logs/check_all_resident_20261124.txt`; `python scripts/board_monitor.py --serial COM4 --resident --check all` |
| 2 | Chu kỳ tính toán mỗi nhịp | **367 091** = **7,34 ms** @ 50 MHz | `cycles` trong `RESULT` của `RUN`, `layer0-5_rec101_93600_beat000_20261124-*.raw.log` (2 lần chạy, y hệt) |
| 3 | Chu kỳ từng layer | conv0 21 429 · conv1 104 379 · conv2 124 603 · conv3 113 321 · fc1 3 137 · fc2 222 | như dòng 2; conv0–conv3 = 363 732, **trùng từng chu kỳ** với mô phỏng RTL (`docs/issue/L04`) |
| 4 | Trước khi sửa loader | 2 058 467 chu kỳ = 41,17 ms | `layer0-5_rec101_93600_beat121_20261118-182057.log`, `multi_all_20261118-165801.log` (bitstream cũ, 921 600 baud, cũng bit-exact 1 306/1 306) |
| 5 | Thông lượng, hiệu suất PE | 0,44 GOPS trên đỉnh 4,9 GOPS = **≈ 8,9 %** (trước: 0,078 GOPS, 1,6 %) | 1 609 344 MAC × 2 / 7,34 ms; 1 MAC = 2 phép |
| 6 | Đường truyền mỗi nhịp (1,5 Mbaud) | gửi lại trọng số từng layer: ≈ 1,0 s, 232 gói, 27 228 B. Nạp một lần (`--resident`): **≈ 0,10 s**, 27 gói, 954 B. Lần nạp ban đầu: 194 gói, 26 195 B, 0,88 s | `check_all_resident_20261124.txt` (link 0,10 s/nhịp); thử 6 nhịp bằng script tạm |
| 7 | Tài nguyên | 41 353 LE (83 %), 15 697 thanh ghi, 452 390 bit nhớ (27 %), 63 M9K, 8/288 nhân 9 bit, 4 chân; mảng PE `conv1d_tile_top` 27 313 LE (66 %) | `docs/quartus/cnn1d.fit.summary` (21/11/2026) |
| 8 | Thời gian | Fmax 50,21 MHz (85 °C), 54,6 MHz (0 °C); slack setup +0,082 ns, hold +0,088 ns, TNS = 0 | `docs/quartus/cnn1d.sta.summary` |
| 9 | Công suất (**ước lượng**) | 332,17 mW = động 232,38 + tĩnh 91,03 + I/O 8,76; độ tin cậy **thấp** (vectorless 12,5 %, không có `.vcd`) | `docs/quartus/cnn1d.pow.summary` (24/11/2026) |
| 10 | Năng lượng mỗi nhịp | ≈ 2,44 mJ (332,17 mW × 7,34 ms); phần động ≈ 1,71 mJ; ≈ 1,3 GOPS/W (phần động ≈ 1,9) | tính từ dòng 2 và 9. **Chỉ tính trên thời gian tính toán**; thiết kế không có clock gating nên công suất gần như không đổi giữa các nhịp |
| 11 | Phân loại (DS2, chạy một lần) | macro-F1 **0,603**; SVEB Se 71,4 %; VEB Se 92,3 %; chênh float–int ≈ 0,0007 | `docs/final_ds2_results.md` |
| 12 | Phát hiện đỉnh R | Se 99,753 %, +P 99,682 % | `docs/system_overview.md` |

## 3. Chưa đo, không được viết thành khẳng định
- Công suất thật (chỉ có ước lượng); năng lượng khi tính trên chu kỳ 0,1 s mỗi nhịp.
- Độ trễ từ đỉnh R đến nhãn đo thực trên đường chạy trực tiếp (panel `board_compare` có ước tính: chờ cửa sổ + FPGA).
- Đóng góp riêng của hai sửa đổi loader (pipeline 1 byte/chu kỳ và pool 6 hàng): hai thay đổi được đưa vào cùng lúc.
- Mọi con số về mảng khác 7×7 hoặc bộ nạp đọc rộng, nạp chồng: chỉ là dự báo từ `scripts/tile_cycle_model.py`.

## 4. Cách tái hiện nhanh

```powershell
$env:PYTHONPATH='D:\pylibs'                                                   # nếu torch cài ở ổ khác
python scripts/layer_board_test.py info --port COM4 --baud 1500000            # kiểm tra kết nối, INFO features 0x3F
python host/ecg_layer_host_pc.py COM4 --baud 1500000                          # một nhịp, từng layer, so golden
python scripts/board_monitor.py --serial COM4 --resident --check all          # 1 306 nhịp, ~4,5 phút
python scripts/board_compare.py --record 208 --serial COM4 --resident --speed 2   # cửa sổ trực tiếp + thông số phần cứng
python scripts/tile_cycle_model.py                                            # mô hình chu kỳ (dự báo)
```

## 5. Quy ước khi dùng số này
- Ghi rõ "không gồm UART" khi nói 7,34 ms; ghi rõ chế độ trọng số khi nói thời gian đường truyền.
- 1 306 nhịp thuộc DS1 (mô hình đã thấy khi huấn luyện): chỉ chứng minh phần cứng khớp phần mềm, không chứng minh tổng quát hoá.
- Khi so sánh với công trình khác: luôn kèm giao thức đánh giá, tập lớp và định nghĩa độ trễ (xem `report/DINHVI.md`).
- Nếu sau này đổi bitstream hoặc đo lại, **thêm mốc mới** (v2) chứ không sửa bảng này.
