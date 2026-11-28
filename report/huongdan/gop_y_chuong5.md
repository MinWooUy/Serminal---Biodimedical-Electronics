# Góp ý Chương 5 (gửi Dung)

Đối chiếu với RTL, code host và mô hình FPGA phần mềm trên `dev` commit `6d492b2`.
Số liệu giao thức (baud, khung, CRC, bảng lệnh, mã NACK, số gói, số byte, tài nguyên) đều khớp.
Dưới đây chỉ là các chỗ cần sửa.

Cách làm:

```powershell
git switch dev
git pull
git switch -c docx/c5-sua
```

Đường dẫn file: `report/chapters/ch05_giao_thuc_uart/`.

## A. Cần sửa

### 1. `s05_trinhtu.tex`, dòng 20: reset board

Câu hiện tại nói "tham số trong RAM bị xoá". Reset **không** xoá RAM; nó đưa con trỏ bank về 0 và xoá cấu hình.
Câu thay thế:

```latex
Nút reset của board không xoá nội dung RAM nhưng đưa con trỏ chọn bank về $0$ và xoá cấu hình layer.
Nếu reset xảy ra giữa hai nhịp, host có thể không nhận lỗi nào trong khi các layer đọc nhầm bank trọng số, nên sau khi reset cần nạp lại mô hình (khởi động lại chương trình host).
Ngược lại, sau một lỗi giao thức, host đánh dấu mô hình không còn hợp lệ và tự nạp lại ở nhịp kế tiếp.
```

Ghi chú: hành vi này suy từ RTL và thử trên mô hình phần mềm, chưa thử trên board.

### 2. `s04_fsmuart.tex`, dòng 30: `recover()`

`recover()` không chạy lại nhịp; nó chỉ làm sạch đường truyền. Thay từ "Thay vào đó, ..." đến hết đoạn bằng:

```latex
Thay vào đó, phương thức \texttt{BoardClassifier.recover()} đưa đường truyền về trạng thái sạch: đánh dấu mô hình thường trú không còn hợp lệ, ngừng gửi $200$~ms (gấp đôi ngưỡng timeout $100$~ms của \texttt{packet\_rx}) và xả bộ đệm cổng nối tiếp.
Sau đó chương trình gọi (\texttt{board\_eval.py}, \texttt{board\_monitor.py}, \texttt{board\_compare.py}) gửi lại nhịp lỗi từ đầu theo tham số \texttt{--retries}, bắt đầu bằng \texttt{WRITE\_X} và \texttt{SWAP}, nên không dùng lại trạng thái dở dang nào.
```

### 3. `s06_phiahost.tex`, dòng 13 và 18: công cụ dòng lệnh

`layer_board_test.py` chỉ có `--port`, không có `--sim`/`--serial`.
Thay câu mở đầu danh sách (dòng 13):

```latex
Bốn công cụ dòng lệnh trong \texttt{scripts/} dùng chung lớp \texttt{LayerHost}; ba công cụ \texttt{board\_monitor.py}, \texttt{board\_compare.py} và \texttt{board\_eval.py} chọn được board thật (\texttt{--serial}) hoặc mô hình giả lập (\texttt{--sim}), còn \texttt{layer\_board\_test.py} chỉ chạy trên board qua \texttt{--port}.
```

Ở mục `board_eval.py` (dòng 18), tên file xuất ra là `beats_<seg>.csv`, thay `beats.csv` bằng:

```latex
\texttt{beats\_detector.csv}, \texttt{beats\_annotation.csv}
```

### 4. Hình 5.2 (`report/figures/ch05_giao_thuc_uart/fsm_cmd.png`)

- `RD_FMAP_ISSUE` → `RD_FMAP_WAIT` là chuỗi nối tiếp, `WAIT` quay lại `ISSUE` cho byte kế tiếp, xong mới sang `SEND`.
- `RUN_START` → `RUN_WAIT` → `SEND` cũng nối tiếp.
- Thêm đường `IDLE` → `SEND` (dùng cho `CONFIG`, `INFO` và mọi `NACK`).
- Thêm đường `TX_WAIT` → `IDLE`.
- Nhãn nhánh `WRITE` thêm `WRITE_SHIFT`.
- Đổi `RD-GAP` thành `RD_GAP`.
- Xuất lại ảnh để mất các dấu × màu xanh trên khung.

### 5. Hình 5.3 (`report/figures/ch05_giao_thuc_uart/trinh_tu.png`)

- Sửa "PFGA" thành "FPGA".
- Bước 1 là nạp nhịp: `WRITE_X`, `SWAP` → `ACK` (không phải "chạy 1 layer").
- Bước 2, lặp cho từng layer: `WRITE_SHIFT`, `CONFIG` → `ACK`; `RUN` → `RESULT`.
- Bước 3: `SWAP` → `ACK`; `READ_FMAP` → `RESULT` 17 byte.

## B. Nên chỉnh, mức nhỏ

| File, dòng | Hiện tại | Đề xuất |
|---|---|---|
| `s05_trinhtu.tex`, 46 | truyền trên dây "$6$--$8$ ms" | khoảng $8{,}4$ ms cho $1\,257$ byte hai chiều |
| `s05_trinhtu.tex`, 46 | trễ khứ hồi "$\approx3{,}5$ ms mỗi gói" | khoảng $3$ ms mỗi gói (Mục 7.5 đã sửa thành 3 ms trên `dev`, commit `7dcf96a`) |
| `s06_phiahost.tex`, 8 | "một nhịp bất kỳ" | "một nhịp bất kỳ trong bộ golden" |
| `s06_phiahost.tex`, 16 | "theo tiêu chuẩn Cổng kiểm chứng 4 (GATE 4)" | bỏ cụm này; nhãn GATE 4 thuộc `live_monitor.py` |
| `s01_uartvl.tex`, 9 | "độ phân giải tối thiểu $\pm4$ chu kỳ" | "để từ cạnh bit start đến giữa bit có ít nhất $4$ chu kỳ xung nhịp" |

Commit message không thêm dòng ghi nhận AI. Xong thì mở Pull Request vào `dev`.
