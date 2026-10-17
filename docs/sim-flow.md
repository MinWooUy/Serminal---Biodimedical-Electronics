# Chạy testbench RTL không cần mở Quartus

Cập nhật: 17/10/2026. Áp dụng cho `rtl/cnn1d/`.

Toàn bộ flow chỉ dùng **ModelSim – Intel FPGA Edition** (cài kèm Quartus Lite) qua script [`rtl/cnn1d/sim/run_tb.sh`](../rtl/cnn1d/sim/run_tb.sh). Không cần mở Quartus và không cần tạo project. Có thể chạy từ terminal của VS Code hoặc từ các task có sẵn trong [`.vscode/tasks.json`](../.vscode/tasks.json).

## Lệnh nhanh

| Việc cần làm | Lệnh (Git Bash, đứng ở gốc repo) | Task trong VS Code |
| --- | --- | --- |
| Chạy 1 testbench | `rtl/cnn1d/sim/run_tb.sh tb_ring_buffer` | RTL: chạy test theo tên |
| Chạy test cho module đang sửa | `rtl/cnn1d/sim/run_tb.sh ring_buffer` | RTL: chạy test cho file đang mở |
| Chạy toàn bộ regression | `rtl/cnn1d/sim/run_tb.sh all` | RTL: chạy toàn bộ regression (`Ctrl+Shift+P` → *Tasks: Run Test Task*) |
| Ghi waveform để xem sau | `rtl/cnn1d/sim/run_tb.sh --wave tb_ring_buffer` | RTL: ghi waveform cho file đang mở |
| Mở ModelSim GUI, xem wave ngay | `rtl/cnn1d/sim/run_tb.sh --gui tb_ring_buffer` | RTL: mở ModelSim GUI cho file đang mở |
| Xóa toàn bộ kết quả, biên dịch lại từ đầu | `rm -rf rtl/cnn1d/sim/out` | — |

Kết quả của mỗi testbench là một dòng: `tb_ring_buffer: PASS=15 FAIL=0 (log: sim/out/tb_ring_buffer.log)`. Script trả mã thoát **0** khi mọi testbench PASS và **1** khi có lỗi biên dịch, lỗi mô phỏng hoặc dòng FAIL.

## 1. Cài đặt một lần

1. **ModelSim**: script mặc định dùng `C:/intelFPGA_lite/18.1/modelsim_ase/win32aloem`. Nếu cài ở chỗ khác, xem trường hợp 8.
2. **Git for Windows**: cung cấp Git Bash tại `C:\Program Files\Git\bin\bash.exe`.
3. **Chọn Git Bash làm terminal trong VS Code:** `Ctrl+Shift+P` → *Terminal: Select Default Profile* → **Git Bash**, rồi mở terminal mới (`` Ctrl+` ``).

> **Không gõ `bash` trong PowerShell hay cmd.** Trên máy này, `bash` trong PATH là `C:\Windows\system32\bash.exe`, tức bash của **WSL**. Nó không thấy đường dẫn `/c/intelFPGA_lite/...` nên script sẽ lỗi. Phải dùng Git Bash, hoặc gọi đầy đủ đường dẫn như ở trường hợp 7.

## 2. Các trường hợp

### Trường hợp 1: chạy một testbench

```bash
rtl/cnn1d/sim/run_tb.sh tb_conv1d_tile_top
```

Script biên dịch mọi file trong [`filelist.f`](../rtl/cnn1d/filelist.f) cùng `tb/<tên>.v` vào `sim/out/work`, rồi chạy `vsim` ở chế độ dòng lệnh cho tới `$finish`. Log nằm ở `rtl/cnn1d/sim/out/<tên>.log`.

### Trường hợp 2: đang sửa một module, muốn test ngay

Mở file module trong VS Code (ví dụ `mem/ring_buffer.v`), rồi chạy task **RTL: chạy test cho file đang mở**. Script tự tìm `tb/tb_ring_buffer.v` theo quy ước tên `tb_<module>.v`. Nếu đang mở chính file testbench thì nó chạy luôn testbench đó.

Từ terminal:

```bash
rtl/cnn1d/sim/run_tb.sh ring_buffer
```

Nếu báo `Không tìm thấy testbench`, tức là module đó chưa có testbench riêng. Xem trường hợp 6.

### Trường hợp 3: trước khi commit

```bash
rtl/cnn1d/sim/run_tb.sh all
```

Lệnh này chạy mọi file `tb/*.v` và kết thúc bằng `REGRESSION: tất cả PASS` hoặc `REGRESSION: N testbench lỗi`. Chỉ commit khi kết quả là tất cả PASS. Các testbench trong `tb/stale/` không được chạy (xem mục 4).

Có thể gắn lệnh này vào git hook `pre-commit` để chặn tự động, vì script trả mã thoát 1 khi có lỗi.

### Trường hợp 4: test FAIL, cần xem waveform

**Cách A: mở GUI, xem ngay**

```bash
rtl/cnn1d/sim/run_tb.sh --gui tb_ring_buffer
```

ModelSim GUI mở ra, đã thêm mọi tín hiệu vào cửa sổ Wave và chạy tới `$finish`. Đóng cửa sổ thì terminal trở lại. Trong GUI, sửa RTL xong muốn chạy lại thì gõ ở ô Transcript:

```tcl
vlog -work sim/out/work -f filelist.f tb/tb_ring_buffer.v
restart -f
run -all
```

**Cách B: ghi waveform ra file, mở sau**

```bash
rtl/cnn1d/sim/run_tb.sh --wave tb_ring_buffer
vsim -view rtl/cnn1d/sim/out/tb_ring_buffer.wlf
```

Chế độ `--wave` chạy không có GUI nhưng ghi mọi tín hiệu (`log -r /*`) vào `.wlf`. Cách này hợp khi chạy một test dài rồi mới xem lại. Chế độ thường không ghi waveform, nên chạy nhanh hơn.

Muốn xem đúng lúc lỗi xảy ra: dòng `[FAIL]` trong log có in thời điểm `t=...`, nhập thời điểm đó vào ô thời gian của cửa sổ Wave.

### Trường hợp 5: lỗi biên dịch

Script in ra 5 dòng `** Error` đầu tiên. Khi chạy bằng task của VS Code, các lỗi này hiện trong khung **Problems** (`Ctrl+Shift+M`); bấm vào sẽ nhảy tới đúng file và dòng. Log đầy đủ nằm ở `sim/out/<tên>.vlog.log`.

| Thông báo | Nguyên nhân thường gặp |
| --- | --- |
| `unexpected non-printable character with the hex value '0xef'` | File lưu dạng **UTF-8 có BOM**. ModelSim 10.5 không đọc được BOM. Trong VS Code: thanh trạng thái → *UTF-8 with BOM* → *Save with Encoding* → **UTF-8** |
| `Register is illegal in left-hand side of continuous assignment` | Dùng `assign` cho biến khai báo `reg`. Đổi thành `wire` |
| `Undefined variable` / `Unresolved reference` | Thiếu file trong `filelist.f`, hoặc sai tên module |
| `Error loading design` / `Could not find 'tb_x'` | Tên module testbench khác tên file, hoặc file chỉ toàn comment |

### Trường hợp 6: thêm module mới hoặc testbench mới

1. Thêm đường dẫn module vào [`filelist.f`](../rtl/cnn1d/filelist.f), đặt vào nhóm thư mục tương ứng (`mem/`, `feeder/`, ...).
2. Testbench đặt tại `tb/tb_<module>.v`, và **tên module testbench phải trùng tên file**.
3. Testbench phải tự kiểm tra kết quả (self-checking) và tuân theo quy ước mà script dùng để đếm:
   - in `[PASS] <tên test>` cho mỗi test đạt;
   - in `[FAIL] ...` cho mỗi lỗi, kèm thời điểm `$time`;
   - cuối cùng in `RESULT : ALL TESTS PASSED` hoặc `RESULT : FAILED`, rồi gọi `$finish`;
   - có một khối `initial` báo timeout (in `TIMEOUT` rồi `$finish`), để test bị treo vẫn dừng.
4. Lưu file dạng UTF-8 **không BOM**.
5. Chạy `rtl/cnn1d/sim/run_tb.sh <module>` rồi `rtl/cnn1d/sim/run_tb.sh all`.

Mẫu tham khảo: [`tb/tb_ring_buffer.v`](../rtl/cnn1d/tb/tb_ring_buffer.v). Testbench này có mô hình tham chiếu chạy song song với DUT, kiểm tra ở mỗi chu kỳ, và chạy 2 cấu hình tham số cùng lúc.

### Trường hợp 7: không muốn mở Git Bash (đang ở PowerShell hoặc cmd)

Gọi Git Bash bằng đường dẫn đầy đủ.

PowerShell:

```powershell
& "C:\Program Files\Git\bin\bash.exe" rtl/cnn1d/sim/run_tb.sh all
```

cmd:

```bat
"C:\Program Files\Git\bin\bash.exe" rtl/cnn1d/sim/run_tb.sh all
```

Các task trong `.vscode/tasks.json` cũng gọi theo cách này, nên chạy được bất kể terminal mặc định là gì.

### Trường hợp 8: máy khác cài ModelSim ở chỗ khác, hoặc dùng Questa

Đặt biến `MODELSIM` trỏ tới thư mục chứa `vlog.exe` và `vsim.exe`:

```bash
MODELSIM="/d/intelFPGA_lite/20.1/modelsim_ase/win32aloem" rtl/cnn1d/sim/run_tb.sh all
```

Muốn đặt cố định, thêm dòng `export MODELSIM=...` vào `~/.bashrc` của Git Bash. Questa dùng cùng lệnh `vlog`/`vsim`, nên chỉ cần đổi đường dẫn.

### Trường hợp 9: kết quả lạ sau khi đổi tên hoặc xóa module

Thư viện `sim/out/work` vẫn còn giữ module cũ. Xóa đi rồi chạy lại:

```bash
rm -rf rtl/cnn1d/sim/out
rtl/cnn1d/sim/run_tb.sh all
```

Thư mục `sim/out/` đã có trong `.gitignore`, xóa thoải mái.

## 3. Script làm gì

```
run_tb.sh <tên>
  ├─ vlib sim/out/work                     (lần đầu)
  ├─ vlog -f filelist.f tb/<tên>.v         -> sim/out/<tên>.vlog.log
  ├─ vsim -c -do "run -all; quit -f"       -> sim/out/<tên>.log
  │     (--wave: thêm "log -r /*" -> sim/out/<tên>.wlf)
  │     (--gui : vsim -gui, "add wave -r /*; run -all")
  └─ đếm dòng PASS / FAIL trong log -> in kết quả, trả mã thoát
```

Biên dịch toàn bộ `filelist.f` mỗi lần chạy chỉ mất vài giây, nên không cần quản lý phụ thuộc.

## 4. Hiện trạng testbench

| Testbench | Kết quả (17/10/2026) |
| --- | --- |
| `tb_conv1d_addr_gen` | PASS 58 |
| `tb_conv1d_layer_controller` | PASS 95 |
| `tb_conv1d_tile_top` | PASS 41 |
| `tb_conv1d_weight_feeder_top` | PASS 22 |
| `tb_conv1d_weight_loader` | PASS 19 |
| `tb_feed_lane` | PASS |
| `tb_ring_buffer` | PASS (7 test × DEPTH 4 và 16) |
| `tb_tensor_pingpong_buffer` | PASS 17 |
| `tb_tensor_sram` | PASS 8 |
| `tb_wallace_tree_multiplier` | PASS |
| `tb/stale/*` (4 file) | Không chạy: lệch interface hiện tại, đã fail ngay ở thư mục gốc `AI_Chip`. Cần sửa hoặc xóa |
