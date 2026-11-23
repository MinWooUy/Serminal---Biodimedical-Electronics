# Định vị của đề tài (cả nhóm dùng thống nhất)

## Câu định vị (đã chốt)

> **Một nền tảng tham chiếu số nguyên, khớp từng bit, cho phân loại nhịp ECG liên bệnh nhân:
> từ mẫu ADC đến nhãn N/S/V/F, với bộ tăng tốc Conv1D cấu hình bằng lệnh chạy trọn mạng trên FPGA MAX 10
> và quy trình kiểm chứng đầu–cuối trên board.**

Trong LaTeX dùng macro `\dinhvi` (định nghĩa **một chỗ** ở `config/macros.tex`); không gõ lại câu này ở nơi khác để cả bài luôn khớp.

## Vì sao định vị như vậy

- **Điểm mạnh thật của ta là tính kiểm chứng**, không phải tốc độ: khớp từng bit trên 1 306 nhịp; giao thức đánh giá liên bệnh nhân (DS1/DS2, DS2 chạy một lần); độ trễ đầu–cuối được phân rã.
- Về tốc độ và công suất ta chưa mạnh (hiệu suất PE ≈ 1,6 %, 83 % LE, **chưa đo công suất**), nên không đặt khẩu hiệu "nhanh" hay "tiết kiệm".
- Khoảng trống trong tài liệu khớp với định vị này: học sâu trên phần cứng còn ít; ít công trình báo cáo nhất quán độ trễ từ thu nhận đến quyết định; một số công trình chấp nhận mất độ chính xác khi lên phần cứng hoặc đánh giá theo mẫu thay vì theo bệnh nhân.
  (Nguồn phân tích: thư mục `seminar/research`, bản `Tong_hop_nghien_cuu.pdf`.)

## Quy tắc cách nói

| Nên | Không nên |
|---|---|
| "khớp từng bit (bit-exact) với golden trên N nhịp" | "chính xác hơn" khi so với bài khác khác giao thức |
| "độ trễ tính toán 41,17 ms/nhịp (không gồm UART)" | "nhanh hơn CPU" nếu chưa đo |
| "chưa đo công suất" hoặc "ước lượng bằng công cụ, giả thiết …" | nêu số công suất khi chưa có nguồn |
| "cùng bậc" / "không so trực tiếp được" | "tốt hơn" khi khác tập lớp hoặc giao thức |
| mọi bảng so sánh có cột **giao thức đánh giá** và **tập lớp** | trích số của bài khác mà không đọc toàn văn |
| "1 306 nhịp thuộc DS1: chỉ kiểm chứng phần cứng" | dùng 1 306 nhịp để nói về tổng quát hoá |

## Luận điểm và bằng chứng

| # | Luận điểm | Bằng chứng trong repo | Trạng thái |
|---|---|---|---|
| C1 | Pipeline số nguyên tương đương bản float | Cổng (iii): chênh macro-F1 OOF ≈ 0,0007 (`docs/system_overview.md`) | Có sẵn |
| C2 | Bản stream khớp từng bit với golden | `scripts/live_monitor.py --check all`: 8 đoạn, 1 306 nhịp (`docs/uart_protocol.md`) | Có sẵn |
| C3 | Accelerator chạy trọn 6 layer trên board, khớp từng bit | `results/hardware_logs/` | Có sẵn |
| C4 | Liên bệnh nhân: macro-F1 0,603; SVEB Se 71,4 %; VEB Se 92,3 % | `docs/final_ds2_results.md` | Có sẵn |
| C5 | 50 MHz; 83 % LE; 27 % bộ nhớ | Báo cáo Quartus | Có sẵn |
| C6 | Tính toán 41,17 ms/nhịp | Log board, **trước** khi sửa loader | Có sẵn; cần đo lại |
| C7 | Sau sửa loader conv0–conv3 ≈ 7,3 ms | Mô phỏng 363 732 chu kỳ (`docs/issue/L04`) | **Cần đo trên board** |
| C8 | Hiệu suất PE ≈ 1,6 % và nguyên nhân | Tính từ MAC và chu kỳ; `docs/issue/L04` | Có sẵn (tính) |
| C9 | Năng lượng mỗi nhịp | — | **Cần đo** (Quartus Power Analyzer) |
| C10 | int8 là điểm cân bằng | — | **Cần quét bit trên DS1** (không dùng DS2) |
| C11 | Độ trễ R → nhãn 1,1–2,4 s gồm ba thành phần | Giá trị lớn nhất trên 8 đoạn golden | **Cần phân rã** |

Câu nào trong bài không có dòng ở bảng này thì chưa nên viết thành khẳng định.

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
