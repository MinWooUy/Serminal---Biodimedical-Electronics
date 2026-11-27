# Định vị của đề tài (cả nhóm dùng thống nhất)

## Câu định vị (đã chốt)

> **Một nền tảng tham chiếu số nguyên, khớp từng bit, cho phân loại nhịp ECG liên bệnh nhân:
> từ mẫu ADC đến nhãn N/S/V/F, với bộ tăng tốc Conv1D cấu hình bằng lệnh chạy trọn mạng trên FPGA MAX 10
> và quy trình kiểm chứng đầu–cuối trên board.**

Trong LaTeX dùng macro `\dinhvi` (định nghĩa **một chỗ** ở `config/macros.tex`); không gõ lại câu này ở nơi khác để cả bài luôn khớp.

## Vì sao định vị như vậy

- **Điểm mạnh thật của ta là tính kiểm chứng**, không phải tốc độ: khớp từng bit trên 1 306 nhịp (đo lại ngày 24/11 trên bitstream hiện tại); giao thức đánh giá liên bệnh nhân (DS1/DS2, DS2 chạy một lần); độ trễ đầu–cuối được phân rã.
- Về tốc độ và công suất ta chưa mạnh (hiệu suất PE ≈ 9 % sau khi sửa loader, trước đó 1,6 %; 83 % LE; công suất **chỉ là ước lượng** của Quartus, 332 mW, độ tin cậy thấp), nên không đặt khẩu hiệu "nhanh" hay "tiết kiệm". Số đo tốc độ chỉ dùng để mô tả hệ thống, không dùng để so với công trình khác.
- Khoảng trống trong tài liệu khớp với định vị này: học sâu trên phần cứng còn ít; ít công trình báo cáo nhất quán độ trễ từ thu nhận đến quyết định; một số công trình chấp nhận mất độ chính xác khi lên phần cứng hoặc đánh giá theo mẫu thay vì theo bệnh nhân.
  (Nguồn phân tích: thư mục `seminar/research`, bản `Tong_hop_nghien_cuu.pdf`.)

## Quy tắc cách nói

| Nên | Không nên |
|---|---|
| "khớp từng bit (bit-exact) với golden trên N nhịp" | "chính xác hơn" khi so với bài khác khác giao thức |
| "độ trễ tính toán 7,34 ms/nhịp (không gồm UART)" và "đường truyền ≈ 0,1 s/nhịp khi nạp mô hình một lần" | "nhanh hơn CPU" nếu chưa đo; nói "7,34 ms" mà không nói là không gồm UART |
| "công suất ước lượng 332 mW (Quartus Power Analyzer, không dùng vector mô phỏng, độ tin cậy thấp)" | gọi 332 mW là "công suất đo được", hoặc so thẳng với số của bài khác như thể cùng điều kiện |
| "năng lượng ≈ 2,4 mJ/nhịp, tính trên thời gian tính toán 7,34 ms" (nói rõ định nghĩa) | nêu năng lượng mỗi nhịp mà không nói là chỉ tính trên thời gian tính toán (thiết kế không có clock gating, công suất gần như không đổi giữa các nhịp) |
| "cùng bậc" / "không so trực tiếp được" | "tốt hơn" khi khác tập lớp hoặc giao thức |
| mọi bảng so sánh có cột **giao thức đánh giá** và **tập lớp** | trích số của bài khác mà không đọc toàn văn |
| "1 306 nhịp thuộc DS1: chỉ kiểm chứng phần cứng" | dùng 1 306 nhịp để nói về tổng quát hoá |

## Luận điểm và bằng chứng

| # | Luận điểm | Bằng chứng trong repo | Trạng thái |
|---|---|---|---|
| C1 | Pipeline số nguyên tương đương bản float | Cổng (iii): chênh macro-F1 OOF ≈ 0,0007 (`docs/system_overview.md`) | Có sẵn |
| C2 | Bản stream khớp từng bit với golden | `scripts/live_monitor.py --check all`: 8 đoạn, 1 306 nhịp (`docs/uart_protocol.md`) | Có sẵn |
| C3 | Accelerator chạy trọn 6 layer trên board, khớp từng bit | `results/hardware_logs/check_all_resident_20261124.txt`: `board_monitor --serial COM4 --resident --check all`, 8 đoạn, **1 306/1 306 nhịp**, digest từng đoạn trùng golden, bitstream hiện tại (`.sof` 21/11, 1,5 Mbaud, INFO features `0x3F`). Bản cũ: `multi_all_20261118-165801.log` (bitstream trước khi sửa loader, 921 600 baud) | Đã đo 24/11 |
| C4 | Liên bệnh nhân: macro-F1 0,603; SVEB Se 71,4 %; VEB Se 92,3 % | `docs/final_ds2_results.md` | Có sẵn |
| C5 | 50 MHz; 83 % LE; 27 % bộ nhớ | Báo cáo Quartus | Có sẵn |
| C6 | Tính toán **7,34 ms/nhịp** (367 091 chu kỳ @50 MHz): conv0 21 429 · conv1 104 379 · conv2 124 603 · conv3 113 321 · fc1 3 137 · fc2 222 | Giá trị `cycles` trong `RESULT` của `RUN`, giải mã từ `layer0-5_rec101_93600_beat000_20261124-085811.raw.log` và `...112417.raw.log` (hai lần chạy cho kết quả y hệt) | Đã đo 24/11 |
| C6' | Trước khi sửa loader: 2 058 467 chu kỳ = 41,17 ms/nhịp | `layer0-5_rec101_93600_beat121_20261118-182057.log`, `multi_all_20261118-165801.log` | Lịch sử; dùng để so "trước/sau" |
| C7 | Sau sửa loader conv0–conv3 = 363 732 chu kỳ (7,27 ms), trùng **từng chu kỳ** với mô phỏng RTL | So log 24/11 với mô phỏng trong `docs/issue/L04` (21 429 / 104 379 / 124 603 / 113 321) | Đã đo 24/11. Hai thay đổi cùng lúc (loader 1 B/chu kỳ + pool 6 hàng) nên chưa tách được đóng góp từng cái bằng số đo |
| C8 | Hiệu suất PE ≈ 8,95 % (1 609 344 MAC theo thuật toán / (367 091 × 49)); trước đó 1,6 %. Nếu tính MAC phần cứng thực hiện (FC1 chạy trên ba hàng) thì là 1 613 696 MAC | Tính từ MAC và chu kỳ ở C6; nguyên nhân còn lại: nạp rồi mới tính, không tái dùng dữ liệu (L04 bước 3, 4 chưa làm) | Có (tính từ số đo) |
| C12 | Nạp mô hình một lần (`--resident`): mỗi nhịp 27 gói / 954 B thay cho 232 gói / 27 228 B; thời gian đường truyền ≈ 0,095–0,10 s/nhịp thay cho ≈ 1,0 s | Thử 6 nhịp trên board bằng script tạm (24/11, không lưu trong repo: 232 → 27 gói, 1,0 → 0,095 s) và `check_all_resident_20261124.txt` (`link 0.10 s/beat`); lần nạp ban đầu 194 gói, 26 195 B, 0,88 s | Đã đo 24/11 (6 nhịp cho 1,0 → 0,095 s; 1 306 nhịp cho 0,10 s/nhịp) |
| C9 | Công suất **ước lượng** 332,17 mW (động 232,38; tĩnh 91,03; I/O 8,76) @50 MHz. Năng lượng ≈ 2,44 mJ/nhịp (332,17 mW × 7,34 ms; riêng phần động ≈ 1,71 mJ) và ≈ 1,3 GOPS/W (phần động ≈ 1,9) | Quartus Power Analyzer 24/11 (`docs/quartus/cnn1d.pow.summary`; báo cáo đầy đủ `rtl/cnn1d/output_files/cnn1d.pow.rpt`, không commit). **Độ tin cậy thấp**: 100 % tín hiệu dùng vectorless với tốc độ chuyển mạch mặc định 12,5 %, không có file mô phỏng. Phần tĩnh chủ yếu (≈ 76 mW) từ nguồn VCCA 2,5 V và VCCD_PLL, cố định của chip | Ước lượng 24/11. Muốn nâng độ tin cậy: nạp `.vcd` từ ModelSim hoặc đo bằng thiết bị USB giữa máy tính và board |
| C10 | int8 là điểm cân bằng | — | **Cần quét bit trên DS1** (không dùng DS2) |
| C11 | Độ trễ R → nhãn 1,1–2,4 s gồm ba thành phần | Giá trị lớn nhất trên 8 đoạn golden | **Cần phân rã** |
| C13 | Board chạy trọn bản ghi 208 (2 946 nhịp do bộ phát hiện tìm ra), nhãn trùng mô hình int8 phần mềm ở cả 2 946 nhịp, 0 nhịp lỗi | `results/2026-11-24_a5fb3f12_board_compare/` (`log.txt`, `metrics.json`: `answered 2946`, `fpga_eq_int8_sw 2946`; thư mục chưa commit). Bản ghi 208 thuộc DS1 nên chỉ dùng để nói phần cứng khớp phần mềm | Đã đo 24/11 |

Câu nào trong bài không có dòng ở bảng này thì chưa nên viết thành khẳng định.

## Cập nhật 24/11/2026 (số liệu phần cứng mới)

Mốc chốt đầy đủ, kèm định danh bitstream và cách tái hiện: [`../docs/baseline_v1.md`](../docs/baseline_v1.md).

Số liệu cũ trong báo cáo và slide ("41,17 ms/nhịp", "0,85 s/nhịp cả UART", "PE ≈ 1,6 %", "chưa đo lại trên board") đến từ bitstream trước khi sửa loader. Thay bằng:

| Đại lượng | Cũ | Mới (đo 24/11) |
|---|---|---|
| Chu kỳ tính toán mỗi nhịp | 2 058 467 (41,17 ms) | **367 091 (7,34 ms)** |
| Hiệu suất PE | 1,6 % | **≈ 9 %** |
| Gói UART mỗi nhịp (không đọc lại layer trung gian) | 232 gói, 27 228 B | **27 gói, 954 B** (nạp mô hình một lần) |
| Thời gian đường truyền mỗi nhịp (1,5 Mbaud) | ≈ 1,0 s | **≈ 0,10 s** |
| Bit-exact trên board | 1 306/1 306 (bitstream cũ, 921 600 baud) | 1 306/1 306 (bitstream hiện tại, 1,5 Mbaud, `--resident`) |

Điểm nghẽn mới: độ trễ khứ hồi mỗi gói (≈ 3 ms, nhiều gói nhỏ theo kiểu dừng-chờ), không còn là trọng số hay tính toán (7,34 ms chỉ còn khoảng 7 % của 0,1 s).

**Công suất (ước lượng, 24/11):** 332,17 mW tổng, độ tin cậy thấp (vectorless); năng lượng ≈ 2,4 mJ/nhịp chỉ tính trên 7,34 ms tính toán. Thiết kế không có clock gating nên công suất gần như không đổi giữa các nhịp: nếu tính trên chu kỳ 0,1 s mỗi nhịp thì năng lượng lớn hơn nhiều (cỡ chục mJ). So với P02 (67,74 mW, 63,48 GOPS/W cùng 50 MHz, Cyclone V): ta cao hơn khoảng 5 lần về công suất và thấp hơn rất nhiều về GOPS/W vì dùng chung một mảng PE (0,44 GOPS) dựng bằng 41 353 LE; chỉ nêu như mốc tham khảo, không kết luận hơn kém.

**Chưa đo:** công suất thật (cần thiết bị đo USB hoặc `.vcd`); thời gian mỗi nhịp trên đường chạy thật (không đọc lại) khi tải tiền xử lý đồng thời; đóng góp riêng của từng bước sửa loader.

**Còn mặc định cũ:** `BoardClassifier` mặc định vẫn gửi lại trọng số cho từng layer; chế độ nạp một lần phải bật bằng `--resident`. Khi viết báo cáo nói rõ đây là một chế độ cấu hình của host, không phải thay đổi RTL.

## Định vị đã được áp vào đâu

| Nơi | Thay đổi |
|---|---|
| `config/macros.tex` | Macro `\dinhvi` |
| `frontmatter/tomtat.tex` | Mục tiêu, khoảng trống, kết luận theo định vị |
| `chapters/ch01_gioi_thieu` | 1.2 (khoảng trống), 1.3 (đoạn "Định vị của đề tài" + câu hỏi nghiên cứu), 1.5 (đóng góp gắn với cách đo) |
| `chapters/ch07_thuc_nghiem` | 7.5 (cách nói về tốc độ), 7.6 (hạn chế, so sánh có cảnh báo) |
| `chapters/ch08_ket_luan` | 8.1 mở bằng `\dinhvi` |
| Slide (`BaoCao/ECG_CNN1D_FPGA.pptx`) | Slide 1, 4, 5, 6, 22, 23, 24 và ghi chú nói |

## Đổi định vị

Nếu nhóm quyết định đổi hướng: sửa câu trong `config/macros.tex` (báo cáo), sửa lại ba nơi trên slide (slide 1 phụ đề, slide 5 thẻ định vị, slide 24 kết luận) và cập nhật file này.
