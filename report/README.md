# Báo cáo (LaTeX): Phân loại nhịp ECG bằng CNN INT8 trên FPGA

Báo cáo của cả nhóm nằm trong thư mục này, viết bằng LaTeX theo mẫu của khoa
(`Mau Bao cao De tai tot nghiep_2025.docx`): A4, lề 3 / 2,5 / 3 / 2 cm, Times New Roman 13 pt, giãn dòng 1,5, tài liệu tham khảo IEEE.

- **Ai làm phần nào, tiến độ:** [`PHANCONG.md`](PHANCONG.md)
- **Thuật ngữ dùng thống nhất:** [`THUATNGU.md`](THUATNGU.md)

## 0. Cài đặt (cả nhóm dùng cùng một bộ công cụ)

Bản đã kiểm thử trên máy của Huy (Windows 10):

| Thành phần | Phiên bản |
|---|---|
| Bản phân phối LaTeX | **MiKTeX 25.12** (cài ở chế độ *user*, không cần quyền admin) |
| Trình biên dịch | **XeLaTeX** (MiKTeX-XeTeX 4.16) — bắt buộc, vì dùng font hệ thống và tiếng Việt Unicode |
| Tài liệu tham khảo | **BibTeX** 4.2 với kiểu `IEEEtran` (không dùng biber) |
| Font | **Times New Roman** (file `times*.ttf` có sẵn trong `C:\Windows\Fonts`) |
| Soạn thảo (khuyến nghị) | VS Code + tiện ích **LaTeX Workshop** (hoặc Overleaf) |

Cách cài cho thành viên mới:

1. Tải **MiKTeX** (Basic MiKTeX Installer) tại <https://miktex.org/download>, cài mặc định; ở mục *Install missing packages on-the-fly* chọn **Yes**
   (lần biên dịch đầu tiên sẽ tự tải các gói còn thiếu, cần có mạng; có thể mất vài phút).
2. Mở **MiKTeX Console** → *Updates* → cập nhật một lần (hết thông báo "have not checked for MiKTeX updates").
3. Kiểm tra trong PowerShell: `xelatex --version` phải in ra `MiKTeX-XeTeX ... (MiKTeX 25.x)`.
4. Chạy `.\build.ps1` trong thư mục `report/`; ra `build/main.pdf` là xong.

macOS/Linux: cài TeX Live (hoặc MacTeX) bản đầy đủ rồi dùng `xelatex` + `bibtex` như dưới; máy không có Times New Roman thì đổi `\setmainfont` trong `config/preamble.tex` sang `TeX Gyre Termes` (chỉ trên máy đó, không commit).

VS Code + LaTeX Workshop: thêm công thức biên dịch XeLaTeX → BibTeX → XeLaTeX ×2 với đối số `-output-directory=build`, hoặc đơn giản là chạy `.\build.ps1` trong terminal.

## 1. Biên dịch

Cần XeLaTeX + BibTeX và font Times New Roman (có sẵn trên Windows).

```powershell
cd report
.\build.ps1            # đầy đủ: xelatex, bibtex, xelatex ×2  → build/main.pdf
.\build.ps1 -Fast      # 1 lượt xelatex khi đang soạn (mục lục / trích dẫn có thể chưa cập nhật)
.\build.ps1 -Open      # mở PDF sau khi biên dịch
.\build.ps1 -Clean     # xoá build/
```

Không có TeX trên máy? Dùng Overleaf: tải thư mục `report/` lên, đặt *Main document* = `main.tex`, *Compiler* = **XeLaTeX**.
(Overleaf không có Times New Roman; đổi tạm `\setmainfont` trong `config/preamble.tex` sang `TeX Gyre Termes` khi biên dịch ở đó.)

## 2. Cấu trúc thư mục

```
report/
├── main.tex                  Khung tài liệu — KHÔNG viết nội dung vào đây
├── config/
│   ├── metadata.tex          Tên đề tài, thành viên, GVHD, tháng/năm; bật/tắt khung gợi ý (sửa MỘT lần, cả nhóm dùng chung)
│   ├── preamble.tex          Gói lệnh và định dạng theo mẫu khoa (ít khi sửa)
│   └── macros.tex            Macro dùng chung: goiy, hinhcho, bangcho, todo, cmt
├── frontmatter/              Bìa, cam đoan, cảm ơn, tóm tắt, viết tắt…
├── chapters/
│   └── chNN_ten/             Mỗi chương một thư mục
│       ├── 00_chuong.tex     Tiêu đề chương + mục đích; chỉ \input các mục
│       └── sNN_ten.tex       MỖI MỤC MỘT FILE — đây là đơn vị phân công
├── appendices/               Phụ lục A–E
├── backmatter/               Danh mục công trình của tác giả
├── bib/references.bib        Tài liệu tham khảo (BibTeX)
└── figures/
    ├── chNN_ten/             Hình của từng chương
    └── src/                  File nguồn của hình (draw.io, svg, …)
```

Các chương: 1 Giới thiệu · 2 Cơ sở và công trình liên quan · 3 Tổng quan hệ thống · 4 Mô hình CNN · 5 Giao thức UART · 6 Thiết kế phần cứng · 7 Thực nghiệm · 8 Kết luận.

## 3. Quy trình làm việc chung (git)

Chia nhỏ theo **file mục** để nhiều người sửa song song mà không xung đột.

1. Xem [`PHANCONG.md`](PHANCONG.md), nhận mục và đánh dấu `◐` đang viết.
2. Tạo nhánh từ `dev`: `git switch dev; git pull; git switch -c report/<tên>-<chương>` (ví dụ `report/huy-ch06`).
3. Chỉ sửa file thuộc mục mình nhận. Cần sửa file người khác (config, macro, bib dùng chung) thì báo trong nhóm trước.
4. Commit nhỏ, thường xuyên; thông điệp theo dạng `report: ch06 viết mục pipeline từng stage`.
5. Trước khi push: `git pull --rebase origin dev`, rồi `.\build.ps1 -Fast` để chắc chắn vẫn biên dịch được. **Không push khi bản dựng lỗi.**
6. Push nhánh, tạo Pull Request vào `dev`; một người khác xem lại rồi merge.

**Không commit:** thư mục `build/`, file `.aux/.log/.pdf` (đã có trong `.gitignore`). Muốn gửi bản PDF thì gửi qua nhóm chat.

## 4. Quy ước viết

- **Mỗi câu một dòng** trong file `.tex` (xuống dòng không ảnh hưởng PDF). Git so sánh theo dòng nên diff gọn, ít xung đột.
- **Nhãn (label) có tiền tố và mã chương**, tránh trùng giữa các thành viên:
  `ch:`, `sec:`, `fig:`, `tab:`, `eq:`; ví dụ `fig:ch6-khoi`, `tab:ch7-ds2`, `eq:ch4-zscore`. Tham chiếu: `Hình~\ref{fig:ch6-khoi}`, `Bảng~\ref{tab:ch7-ds2}`.
- **Hình:** đặt trong `figures/chNN_ten/`; ưu tiên PDF/SVG vector, ảnh chụp dưới ~1 MB; chữ trong hình bằng cỡ chữ báo cáo; hình lấy từ nguồn khác phải trích dẫn nguồn trong chú thích. Tên bảng ở trên bảng, tên hình ở dưới hình (theo mẫu).
- **Trích dẫn:** thêm mục mới vào cuối `bib/references.bib`, khoá dạng `tacgia+nam+tukhoa` (ví dụ `pantompkins1985realtime`). Viết `Nghiên cứu của Pan và Tompkins~\cite{...} cho thấy…`, không viết `Nghiên cứu~\cite{...} cho thấy…`.
- **Số liệu:** mỗi con số phải truy được về nguồn (`docs/`, `results/`, log, báo cáo Quartus); ghi nguồn vào chú thích hoặc `% nguồn: …`.
- **Viết tắt:** lần đầu viết đầy đủ kèm viết tắt trong ngoặc; thêm vào `frontmatter/viettat.tex` nếu dùng nhiều lần.
- **Ghi chú trong nhóm:** `\todo{việc cần làm}` (chữ đỏ) và `\cmt{Tên}{nội dung}` (chữ xanh) chỉ hiện khi `\showhintstrue`.
  Khung xám **"Dàn ý — nội dung cần điền"** (môi trường `goiy`) là gợi ý ban đầu: viết nội dung xong thì xoá khung.
- Thuật ngữ theo [`THUATNGU.md`](THUATNGU.md).

## 5. Trước khi nộp

- [ ] Đặt `\showhintsfalse` trong `config/metadata.tex` (ẩn mọi khung gợi ý, `\todo`, `\cmt`).
- [ ] Xoá dòng `\nocite{*}` trong `main.tex`; kiểm tra mọi `\cite` đều có mục trong `bib/references.bib`.
- [ ] Điền `config/metadata.tex` (tên đề tài, thành viên, MSSV, GVHD, tháng/năm).
- [ ] Biên dịch đầy đủ, không còn cảnh báo `undefined` trong `build/main.log`.
- [ ] Nội dung chính 50–100 trang (không tính bìa, mục lục, tài liệu tham khảo), theo mẫu khoa; trang nhận xét GVHD/GVPB đã chèn bản có chữ ký.
- [ ] Đối chiếu mọi số liệu với bản mới nhất của `docs/`, `results/`.
