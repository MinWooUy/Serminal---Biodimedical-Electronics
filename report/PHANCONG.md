# Phân công và tiến độ viết báo cáo

Chương 1–3 đã chốt (Long phụ trách chính). Các chương còn lại đang để theo *đề xuất* dựa trên phần việc mỗi người đã làm trong dự án; chỉnh lại cho đúng rồi commit.
Mỗi mục là **một file** `.tex`, nên mỗi người chỉ sửa file của mình và không đụng nhau khi merge.

## Thành viên

| Tên | Vai trò trong dự án | Phần báo cáo |
|---|---|---|
| Long | Mới vào nhóm, đọc source để viết | **Chính: chương 1, 2, 3** |
| Huy | RTL accelerator + host | Hỗ trợ chương 1, 2; chính chương 6 (đề xuất) |
| Hùng (git: `zhunngg`) | Mô hình, huấn luyện, kết quả (python/, results/, export/) | Hỗ trợ chương 3; chính chương 4, phụ lục B, D (đề xuất) |
| Dung | UART, GVU, pipeline controller | Chương 5, phụ lục A (đề xuất) |

Nguyên tắc: người **phụ trách chính** viết và commit mục đó; người **hỗ trợ** cung cấp thông tin, góp ý và rà soát (không sửa trực tiếp file của người khác nếu chưa báo).
Tài liệu nguồn để Long đọc cho chương 1–3: [`HUONGDAN_CH1-3.md`](HUONGDAN_CH1-3.md).

Trạng thái: `☐` chưa viết · `◐` đang viết · `☑` xong bản nháp · `✔` đã rà soát.

## Chương 1. Giới thiệu

Thư mục: `chapters/ch01_gioi_thieu/` — phụ trách chính: **Long** · hỗ trợ: **Huy**

| File | Mục | Phụ trách | Hỗ trợ | Trạng thái |
|---|---|---|---|---|
| `s01_boicanh.tex` | Bối cảnh và lý do chọn đề tài | Long | Huy | ☐ |
| `s02_baitoan.tex` | Phát biểu bài toán | Long | Huy | ☐ |
| `s03_muctieu.tex` | Mục tiêu và câu hỏi nghiên cứu | Long | Huy | ☐ |
| `s04_phamvi.tex` | Đối tượng và phạm vi nghiên cứu | Long | Huy | ☐ |
| `s05_donggop.tex` | Đóng góp của đề tài | Long | Huy | ☐ |
| `s06_bocuc.tex` | Bố cục báo cáo | Long | Huy | ☐ |

## Chương 2. Cơ sở lý thuyết và các công trình liên quan

Thư mục: `chapters/ch02_co_so_lien_quan/` — phụ trách chính: **Long** · hỗ trợ: **Huy**

| File | Mục | Phụ trách | Hỗ trợ | Trạng thái |
|---|---|---|---|---|
| `s01_ecg.tex` | Tín hiệu ECG và phân loại nhịp | Long | Huy | ☐ |
| `s02_qrs.tex` | Phát hiện QRS và tiền xử lý | Long | Huy | ☐ |
| `s03_cnn1d.tex` | Mạng nơ-ron tích chập 1D và lượng tử hoá | Long | Huy | ☐ |
| `s04_accel.tex` | Bộ tăng tốc suy luận trên phần cứng | Long | Huy | ☐ |
| `s05_lienquan.tex` | Các công trình liên quan và khoảng trống nghiên cứu | Long | Huy | ☐ |

## Chương 3. Tổng quan hệ thống đề xuất

Thư mục: `chapters/ch03_tong_quan_he_thong/` — phụ trách chính: **Long** · hỗ trợ: **Hùng**

| File | Mục | Phụ trách | Hỗ trợ | Trạng thái |
|---|---|---|---|---|
| `s01_yeucau.tex` | Yêu cầu và ràng buộc thiết kế | Long | Hùng | ☐ |
| `s02_kientruc.tex` | Kiến trúc tổng thể | Long | Hùng | ☐ |
| `s03_hwsw.tex` | Phân chia phần cứng/phần mềm | Long | Hùng | ☐ |
| `s04_luongdl.tex` | Luồng dữ liệu của một nhịp | Long | Hùng | ☐ |
| `s05_kiemchung.tex` | Chiến lược kiểm chứng | Long | Hùng | ☐ |

## Chương 4. Mô hình CNN và pipeline phần mềm

Thư mục: `chapters/ch04_mo_hinh_cnn/` — phụ trách chính: **Hùng** · hỗ trợ: —

| File | Mục | Phụ trách | Hỗ trợ | Trạng thái |
|---|---|---|---|---|
| `s01_dulieu.tex` | Dữ liệu và giao thức đánh giá | Hùng | — | ☐ |
| `s02_tienxuly.tex` | Tiền xử lý số nguyên | Hùng | — | ☐ |
| `s03_cnnlite.tex` | Kiến trúc CNN Lite | Hùng | — | ☐ |
| `s04_huanluyen.tex` | Huấn luyện | Hùng | — | ☐ |
| `s05_luongtu.tex` | Lượng tử hoá int8 | Hùng | — | ☐ |
| `s06_golden.tex` | Golden vectors | Hùng | — | ☐ |

## Chương 5. Giao thức giao tiếp Host--Accelerator

Thư mục: `chapters/ch05_giao_thuc_uart/` — phụ trách chính: **Dung** · hỗ trợ: —

| File | Mục | Phụ trách | Hỗ trợ | Trạng thái |
|---|---|---|---|---|
| `s01_uartvl.tex` | Tầng vật lý UART | Dung | — | ☐ |
| `s02_khungoi.tex` | Khung gói và CRC-16 | Dung | — | ☐ |
| `s03_bolenh.tex` | Bộ lệnh | Dung | — | ☐ |
| `s04_fsmuart.tex` | Máy trạng thái và xử lý lỗi | Dung | — | ☐ |
| `s05_trinhtu.tex` | Trình tự thực thi một nhịp | Dung | — | ☐ |
| `s06_phiahost.tex` | Phía host | Dung | — | ☐ |

## Chương 6. Thiết kế phần cứng bộ tăng tốc

Thư mục: `chapters/ch06_phan_cung/` — phụ trách chính: **Huy** · hỗ trợ: —

| File | Mục | Phụ trách | Hỗ trợ | Trạng thái |
|---|---|---|---|---|
| `s01_tongquanhw.tex` | Tổng quan kiến trúc | Huy | — | ☐ |
| `s02_mem.tex` | Bộ nhớ tensor ping-pong | Huy | — | ☐ |
| `s03_stage.tex` | Pipeline dữ liệu một layer: từng stage | Huy | — | ☐ |
| `s04_pe.tex` | Mảng PE và số học | Huy | — | ☐ |
| `s05_ctrl.tex` | Điều khiển layer và tile | Huy | — | ☐ |
| `s06_anhxa.tex` | Ánh xạ CNN lên accelerator | Huy | — | ☐ |
| `s07_mophong.tex` | Mô phỏng và kiểm thử RTL | Huy | — | ☐ |
| `s08_bandwidth.tex` | Tối ưu băng thông nạp dữ liệu (nếu đưa vào báo cáo) | Huy | — | ☐ |

## Chương 7. Thực nghiệm, đánh giá và bàn luận

Thư mục: `chapters/ch07_thuc_nghiem/` — phụ trách chính: **Huy + Hùng** · hỗ trợ: —

| File | Mục | Phụ trách | Hỗ trợ | Trạng thái |
|---|---|---|---|---|
| `s01_moitruong.tex` | Môi trường và phương pháp đo | Huy + Hùng | — | ☐ |
| `s02_dochinhxac.tex` | Độ chính xác của mô hình | Huy + Hùng | — | ☐ |
| `s03_bitexact.tex` | Kiểm chứng bit-exact | Huy + Hùng | — | ☐ |
| `s04_taiguyen.tex` | Tài nguyên và thời gian | Huy + Hùng | — | ☐ |
| `s05_hieunang.tex` | Hiệu năng | Huy + Hùng | — | ☐ |
| `s06_banluan.tex` | Bàn luận và đe doạ tính hợp lệ | Huy + Hùng | — | ☐ |

## Chương 8. Kết luận và hướng phát triển

Thư mục: `chapters/ch08_ket_luan/` — phụ trách chính: **Cả nhóm** · hỗ trợ: —

| File | Mục | Phụ trách | Hỗ trợ | Trạng thái |
|---|---|---|---|---|
| `s01_ketqua.tex` | Kết quả đạt được | Cả nhóm | — | ☐ |
| `s02_hanche.tex` | Hạn chế | Cả nhóm | — | ☐ |
| `s03_huongphattrien.tex` | Hướng phát triển | Cả nhóm | — | ☐ |

## Phần đầu, phần cuối, phụ lục

| File | Nội dung | Phụ trách | Hỗ trợ | Trạng thái |
|---|---|---|---|---|
| `config/metadata.tex` | Tên đề tài, thành viên, GVHD, tháng/năm | Cả nhóm | — | ☐ |
| `frontmatter/tomtat.tex` | Tóm tắt 200–300 từ (viết sau cùng) | Cả nhóm | — | ☐ |
| `frontmatter/viettat.tex` | Danh mục ký hiệu, chữ viết tắt | Cả nhóm | — | ☐ |
| `frontmatter/camdoan.tex, loicamon.tex, decuong.tex, nhanxet.tex, thuyetminh.tex` | Các trang đầu theo mẫu khoa | Cả nhóm | — | ☐ |
| `backmatter/congtrinh.tex` | Danh mục công trình của tác giả | Cả nhóm | — | ☐ |
| `appendices/pl_a_bo_lenh.tex` | Phụ lục A: bộ lệnh UART | Dung / Huy | — | ☐ |
| `appendices/pl_b_luong_tu_hoa.tex` | Phụ lục B: tham số lượng tử hoá | Hùng | — | ☐ |
| `appendices/pl_c_tai_nguyen.tex` | Phụ lục C: tài nguyên, timing | Huy | — | ☐ |
| `appendices/pl_d_ket_qua_ds2.tex` | Phụ lục D: kết quả chi tiết DS2 | Hùng | — | ☐ |
| `appendices/pl_e_tai_lap.tex` | Phụ lục E: hướng dẫn tái lập | Cả nhóm | — | ☐ |
| `bib/references.bib` | Tài liệu tham khảo (thêm mục mới ở cuối, đúng khoá) | Mọi người | — | ☐ |
