# Hướng dẫn đọc source để viết chương 1–3

Dành cho **Long** (phụ trách chính chương 1, 2, 3). Huy hỗ trợ chương 1, 2; Hùng hỗ trợ chương 3.
Chương 1–3 không cần đọc RTL từng dòng: chủ yếu là hiểu **bài toán, hệ thống và mô hình** rồi viết lại bằng lời của mình, có nguồn.

## Bước 0 — Lấy source và biên dịch thử

1. Clone repo, chuyển sang nhánh `dev`; thư mục báo cáo là `report/`.
2. Cài LaTeX theo hướng dẫn trong [`README.md`](README.md) mục "Cài đặt", chạy `.\build.ps1` thấy `build/main.pdf` là xong.
3. Mở `PHANCONG.md`, đánh dấu `◐` các mục của mình.

## Bước 1 — Thứ tự đọc (khoảng 2–3 giờ)

| # | Tài liệu | Đọc để biết gì |
|---|---|---|
| 1 | `docs/system_overview.md` | Luồng toàn hệ thống từ mẫu ADC đến nhãn N/S/V/F; hai nhánh phát hiện R và hình dạng; bản float và bản số nguyên |
| 2 | `docs/cnn_explained.md` | CNN Lite từng lớp, vì sao gộp BN, lượng tử hoá int8, vì sao dùng argmax |
| 3 | `docs/final_ds2_results.md` | Kết quả cuối trên DS2 (chạy một lần) |
| 4 | `docs/uart_protocol.md` | Khung gói, CRC, cổng kiểm chứng GATE 4 (đọc phần "Verification") |
| 5 | `docs/spec.pdf` (hoặc `spec.docx`) | Đặc tả hệ thống, đối chiếu với các tài liệu trên |
| 6 | `rtl/cnn1d/de10_lite_top.v` và phần chú thích đầu `rtl/cnn1d/top/layer_cmd_controller.v` | Tham số board và bộ lệnh (chỉ cần hiểu ở mức tổng quan) |
| 7 | `docs/issue/hw-accelerator-limitations.md` | Hạn chế của accelerator (dùng cho phần công trình liên quan và hạn chế) |

Hỏi Huy khi gặp phần RTL khó hiểu; hỏi Hùng khi hỏi về huấn luyện, dữ liệu, kết quả.

## Bước 2 — Nguồn cho từng mục cần viết

### Chương 1. Giới thiệu (`chapters/ch01_gioi_thieu/`)

| Mục | Nguồn trong repo | Việc cần làm |
|---|---|---|
| Bối cảnh, lý do chọn đề tài | `docs/system_overview.md` §1; `docs/spec.pdf` | Tìm 2–3 nguồn ngoài về loạn nhịp và ECG biên (ghi vào `bib/references.bib`) |
| Phát biểu bài toán | `docs/system_overview.md` §3.1 (dữ liệu, nhãn AAMI) | Nêu vào/ra, khó khăn (liên bệnh nhân, mất cân bằng lớp) |
| Mục tiêu, câu hỏi nghiên cứu | `docs/final_ds2_results.md`, `docs/uart_protocol.md` (GATE 4) | Biến thành mục tiêu **đo được** |
| Phạm vi | `docs/system_overview.md` §3.1, `rtl/cnn1d/de10_lite_top.v` | Nêu cả phần *ngoài* phạm vi |
| Đóng góp | `docs/system_overview.md` §2, §5 | Hỏi Huy xác nhận phần nào do nhóm tự làm |
| Bố cục | — | Viết sau cùng |

### Chương 2. Cơ sở lý thuyết và công trình liên quan (`chapters/ch02_co_so_lien_quan/`)

| Mục | Nguồn trong repo | Việc cần làm |
|---|---|---|
| ECG và phân loại nhịp | `docs/system_overview.md` §3.1 | Giải thích ngắn N/S/V/F, MIT-BIH, DS1/DS2 |
| Phát hiện QRS, tiền xử lý | `docs/system_overview.md` §3.2–3.7 | Chỉ nêu **ý tưởng** Pan–Tompkins, lọc, z-score, RR; công thức chi tiết để chương 4 (Hùng) |
| CNN 1D, lượng tử hoá | `docs/cnn_explained.md` §1–5 | Giải thích khái niệm; số liệu cụ thể dành cho chương 4 |
| Bộ tăng tốc phần cứng | `docs/issue/hw-accelerator-limitations.md`, `rtl/cnn1d/comm/README.md` | Nêu systolic array, dataflow ở mức khái niệm |
| Công trình liên quan | Tự tìm | Đọc 4–6 bài, lập bảng so sánh (Bảng trong mục này); **chỉ ghi bài đã đọc** |

Gợi ý tài liệu đã có sẵn trong `bib/references.bib` (kiểm tra lại thông tin trước khi dùng): MIT-BIH, de Chazal, Pan–Tompkins, Kiranyaz, Jacob (lượng tử hoá), Kung (systolic).

### Chương 3. Tổng quan hệ thống đề xuất (`chapters/ch03_tong_quan_he_thong/`)

| Mục | Nguồn trong repo | Việc cần làm |
|---|---|---|
| Yêu cầu, ràng buộc | `rtl/cnn1d/de10_lite_top.v`, `docs/de10_lite/DE10-Lite_User_Manual.pdf` | Nêu chip, clock, bộ nhớ, giao tiếp |
| Kiến trúc tổng thể | `docs/system_overview.md` §1 (sơ đồ mermaid) | Vẽ lại thành hình (đặt trong `figures/ch03_tong_quan_he_thong/`) |
| Phân chia HW/SW | `host/ecg_layer_host_pc.py`, `host/board_classifier.py` | Phối hợp với Hùng |
| Luồng dữ liệu một nhịp | `docs/system_overview.md` §3.6–3.9, `docs/uart_protocol.md` | Vào: 360 mẫu + RR; ra: 4 logit + nhãn |
| Chiến lược kiểm chứng | `docs/uart_protocol.md` mục "Verification and GATE 4" | Chuỗi float → int8 → RTL → board |

## Bước 3 — Chạy thử để hiểu (không cần board)

Các lệnh dưới đây lấy từ `docs/system_overview.md` §5; cần dữ liệu MIT-BIH trong `data/raw/` (tải bằng `scripts/download_mitbih.py`) và môi trường Python của dự án.

```powershell
python scripts/live_monitor.py --check all      # kiểm tra bản stream = golden (không cần board)
python scripts/trace_beat.py --golden rec208_237600 --beat 12   # theo dõi một nhịp qua mọi tầng
python -m pytest tests -q                       # chạy bộ kiểm thử
```

Nếu chưa cài được môi trường Python, nhờ Huy hoặc Hùng chạy hộ và gửi kết quả/hình.

## Lưu ý quan trọng

- **Một số tài liệu không nằm trong git**: `docs/log/`, `docs/drawio/`, `results/img/` (ảnh Quartus, kết quả biên dịch), `model/`, và phần lớn `docs/issue/` (chỉ vài file được commit). Cần gì trong đó thì xin Huy qua nhóm.
- Mọi **con số** (độ chính xác, tài nguyên, thời gian) phải ghi nguồn; nếu không chắc, hỏi người phụ trách chương tương ứng thay vì đoán.
- Không chép nguyên văn từ tài liệu nội bộ hoặc bài báo; viết lại bằng lời của mình và trích dẫn.
- Các khung xám "Dàn ý — nội dung cần điền" trong từng mục là gợi ý của Huy: viết xong thì xoá khung.
