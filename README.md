# Phân loại nhịp ECG bằng CNN1D int8 trên FPGA (DE10-Lite / MAX 10)

Pipeline ECG số nguyên (lọc, Pan-Tompkins, z-score int8) → CNN Lite int8 (6 layer) chạy trên bộ tăng tốc Conv1D
mảng PE 7×7, điều khiển từ máy tính qua UART bằng bộ lệnh CONFIG/RUN.

| Thư mục | Nội dung |
|---|---|
| `python/` | Thư viện: dữ liệu, DSP, mô hình, lượng tử hoá, giao thức UART, mô hình FPGA mềm (`python/link/layersim`) |
| `host/` | Phía máy tính điều khiển board (`ecg_layer_host_pc.py`, `board_classifier.py`) |
| `scripts/` | Công cụ chạy/đánh giá (`live_monitor`, `board_monitor`, `board_eval`, `board_compare`, `layer_board_test`, …) |
| `tests/` | Test Python (pytest) |
| `rtl/cnn1d/` | RTL Verilog, testbench `tb/`, script mô phỏng `sim/`, project Quartus |
| `export/` | `weights.h`, bộ golden (`export/golden`, `golden_manifest.json`) dùng chung cho mọi cách kiểm thử |
| `results/` | Kết quả; log board ở `results/hardware_logs/` |
| `docs/` | Tài liệu: `system_overview.md`, `uart_protocol.md`, `final_ds2_results.md` |
| `report/` | Báo cáo LaTeX của nhóm (xem `report/README.md`) |

Chạy mọi lệnh **từ thư mục gốc của repo**.

---

## Cách chạy test

Có 4 nhóm, đi từ rẻ đến tốn kém. Cùng một bộ golden (`export/golden`) được dùng ở nhóm A và C, nên cùng một nhịp
có thể được so khớp từ phần mềm, mô phỏng RTL đến board.

| Nhóm | Cần gì | Kiểm tra gì |
|---|---|---|
| **A. Phần mềm Python** | Python + thư viện | Pipeline, giao thức, mô hình FPGA mềm khớp golden |
| **B. Mô phỏng RTL** | ModelSim hoặc Icarus | Từng khối RTL và toàn top đúng chức năng |
| **C. Trên board** | DE10-Lite đã nạp bitstream + cáp UART | Phần cứng thật khớp bit với golden |
| **D. Kiểm tra tĩnh** | Quartus | Biên dịch, tài nguyên, thời gian (timing) |

### Dùng trên PowerShell, Git Bash hay Linux

Các khối lệnh bên dưới chỉ gồm `python ...` và đường dẫn dùng dấu `/`, nên chạy được nguyên xi trên PowerShell và Git Bash.
Riêng các script `.sh` ở nhóm B cần bash (Git Bash, WSL hoặc Linux). Trên **Linux/macOS** đổi các điểm sau:

| Chỗ | Windows | Linux / macOS |
|---|---|---|
| Lệnh Python | `python` | `python3` (hoặc `python` khi đã kích hoạt venv); cài gói: `python3 -m pip install ...` |
| Cổng serial | `COM5` | `/dev/ttyUSB0` (cần quyền: `sudo usermod -aG dialout $USER`, rồi đăng nhập lại) |
| ModelSim | mặc định `/c/intelFPGA_lite/18.1/modelsim_ase/win32aloem` | đặt biến `MODELSIM` trỏ tới thư mục chứa `vsim`, ví dụ `MODELSIM=<thư mục cài>/modelsim_ase/linuxaloem sim/run_tb.sh all` |
| Icarus | cài và thêm vào PATH | `sudo apt install iverilog` |

Các ghi chú này suy ra từ việc đọc script, chưa thử trên Linux.

### 0. Chuẩn bị môi trường

```powershell
python -m pip install numpy scipy scikit-learn pyserial wfdb pyyaml pytest matplotlib
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu    # bản CPU, đủ để suy luận
python scripts/download_mitbih.py                          # tải MIT-BIH vào data/raw/mitdb (cần mạng)
```

- Các script `board_monitor`, `live_monitor`, `board_eval`, `board_compare` đều import `wfdb`, `torch`, `yaml` và `sklearn`: thiếu một trong số đó sẽ báo `ModuleNotFoundError` ngay khi khởi động, kể cả khi chạy `--sim`.
- Ổ C: hết chỗ thì `pip install` báo `No space left on device`. Cách tạm: cài sang ổ khác rồi trỏ `PYTHONPATH` tới đó, ví dụ
  `python -m pip install --target D:\pylibs torch scikit-learn`, sau đó trong Git Bash `export PYTHONPATH='D:\pylibs'`
  (PowerShell: `$env:PYTHONPATH='D:\pylibs'`) trước khi chạy script.
- Test gắn dấu `needs_data` sẽ **tự bỏ qua (skip)** nếu thiếu dữ liệu DS1 trong `data/raw/mitdb`; các test còn lại vẫn chạy.
- **DS2 là tập kiểm tra cuối, chỉ chạy một lần.** Đừng đặt `ECG_ALLOW_DS2=1` trừ khi đang làm đánh giá cuối và đã thống nhất trong nhóm.
- Đánh giá trên DS1 chỉ chứng minh FPGA == phần mềm (DS1 đã nằm trong dữ liệu huấn luyện mô hình cuối), **không** dùng để nói về khả năng tổng quát hoá.

### A. Test phần mềm (không cần board)

**A1. Bộ pytest** (24 file trong `tests/`):

```powershell
python -m pytest tests -q
```

Chạy riêng một file hoặc một test:

```powershell
python -m pytest tests/test_layer_protocol.py -q
python -m pytest tests/test_board_classifier.py::test_simulated_board_returns_golden_logits_and_class -q
```

Một số file đáng chú ý: `test_layer_protocol.py` (giao thức lệnh, dùng transport giả), `test_layer_model.py`
(mô hình layer so với golden), `test_board_classifier.py` (mô hình FPGA mềm trả đúng logit và nhãn),
`test_pipeline_int.py` / `test_quant.py` (pipeline số nguyên và lượng tử hoá), `test_pan_tompkins.py`, `test_stream.py`.

**A2. Kiểm tra trọng số FC1/FC2 trong `weights.h` với mọi nhịp golden** (không cần FPGA):

```powershell
python host/ecg_layer_host_pc.py --validate-only
```

**A3. Kiểm tra bit-exact cả đoạn ECG với mô hình FPGA mềm** (không giao diện; so digest cuối với golden):

```powershell
python scripts/board_monitor.py --sim --check rec208_237600      # một đoạn
python scripts/live_monitor.py --check all                       # cả 8 đoạn golden (1 306 nhịp)
```

Tên các đoạn golden: xem `export/golden_manifest.json` (`rec101_93600`, `rec118_237600`, `rec119_489600`,
`rec201_522000`, `rec207_604800`, `rec208_237600`, `rec209_198000`, `rec223_187200`).

**A4. Chấm cả một record bằng mô hình FPGA mềm** (logit FPGA mềm phải khớp phần mềm từng bit):

```powershell
python scripts/board_eval.py --record 208 --sim                  # kết quả vào results/<ngày>_<hash>_board_eval/
python scripts/board_compare.py --record 208 --sim --no-display  # bảng/tệp, không mở cửa sổ
```

### B. Mô phỏng RTL (testbench trong `rtl/cnn1d/tb/`)

**B1. ModelSim** (đã có `vsim` trong PATH trên máy Huy: Intel FPGA Lite 18.1). Chạy trong Git Bash:

```bash
cd rtl/cnn1d
sim/run_tb.sh tb_conv1d_tile_top        # một testbench (có thể gõ tên module: ring_buffer)
sim/run_tb.sh all                       # hồi quy: mọi testbench trong tb/
sim/run_tb.sh --wave tb_ring_buffer     # ghi dạng sóng vào sim/out/<tb>.wlf
sim/run_tb.sh --gui  tb_ring_buffer     # mở ModelSim GUI
sim/run_tb.sh --param ARRAY_ROWS=7 --param ARRAY_COLS=7 tb_conv1d_tile_top
```

Mã thoát 0 khi mọi testbench PASS, 1 khi có lỗi/FAIL. `all` chỉ chạy các file `tb/*.v`; testbench SystemVerilog của GVU (`*.sv`) chạy bằng `run_gvu_tests.sh` (Icarus). Cài ModelSim ở chỗ khác thì đặt biến `MODELSIM`.

**B2. Icarus Verilog** (`iverilog`, `vvp` phải có trong PATH; có thể đặt `IVERILOG`, `VVP`):

```bash
cd rtl/cnn1d
sim/run_comm_tests.sh        # CRC16, UART, packet_rx, uart_crc_top
sim/run_pc_layer_tests.sh    # luồng lệnh PC → FPGA, cnn_pipeline_controller
sim/run_post_tests.sh        # post_process (vector ngẫu nhiên có golden) + conv1d_tile_top
sim/run_gvu_tests.sh         # khối GVU (SystemVerilog)
```

Sinh lại vector FC cho `tb_fc_layers` (script tự kiểm lại với golden trước khi ghi `rtl/cnn1d/sim/hex/fc_*.hex`):

```powershell
python scripts/rtl/gen_fc_hex.py
```

### C. Test trên board DE10-Lite

Chuẩn bị: nạp bitstream; nối USB-UART (TX→RX, RX←TX, chung GND, mức logic tương thích); xác định cổng COM (ví dụ `COM5`).
Baud phải khớp bitstream: `board_monitor.py` và `board_eval.py` mặc định **1 500 000**, `ecg_layer_host_pc.py` mặc định **921 600**; đặt `--baud` cho đúng.

```powershell
# 1) Bring-up tối thiểu: đọc INFO, rồi smoke test
python scripts/layer_board_test.py info  --port COM5
python scripts/layer_board_test.py smoke --port COM5

# 2) Chạy một nhịp golden từng layer (so với golden; log UART vào results/hardware_logs/)
python host/ecg_layer_host_pc.py COM5 --baud 1500000 --record rec101_93600 --beat 0
python host/ecg_layer_host_pc.py COM5 --baud 1500000 --first-layer 0 --last-layer 3     # chỉ conv0..conv3

# 3) Cả đoạn ECG, so digest với golden
python scripts/board_monitor.py --serial COM5 --check rec208_237600

# 3b) Như trên nhưng nạp mô hình MỘT lần (cờ --resident): mỗi nhịp chỉ gửi CONFIG w_base/b_base, nhanh hơn ~10 lần
python scripts/board_monitor.py --serial COM5 --resident --check rec208_237600

# 4) Chấm cả record trên board (thử ngắn trước, rồi mới cả record)
python scripts/board_eval.py --record 208 --serial COM5 --start 600 --dur 60
python scripts/board_eval.py --record 208 --serial COM5
```

Cờ `--resident` có ở `board_monitor.py`, `board_eval.py` và `board_compare.py`. Nó cần bitstream có INFO `features` bit 4
(w_base/b_base; bitstream hiện tại báo `0x3F`). Mặc định vẫn là gửi lại trọng số cho từng layer. Board bị reset hay tắt nguồn
thì dừng và chạy lại tiến trình để mô hình được nạp lại (chương trình tự nạp lại sau mỗi lỗi, nhưng không phát hiện được board vừa reset).

Trước khi chạy trên board, nên chạy A2 và A3 để chắc phần mềm và golden không có vấn đề.
Mỗi nhịp mất khoảng 0,85 s (1,5 Mbaud), vì vậy cả record 30 phút chạy rất lâu.

### D. Biên dịch và thời gian (Quartus)

Mở `rtl/cnn1d/cnn1d.qpf` (Quartus Lite 18.1) và biên dịch; xem báo cáo Fitter (tài nguyên) và TimeQuest (thời gian).
Ràng buộc nằm ở `rtl/cnn1d/SDC1.sdc`. Kết quả hiện tại: 83 % LE, 27 % bộ nhớ, 50 MHz (setup slack +0,082 ns).

### Quy ước khi thêm test

- Test Python mới đặt trong `tests/test_*.py`; test cần dữ liệu MIT-BIH thì gắn `@pytest.mark.needs_data`.
- Testbench mới đặt `rtl/cnn1d/tb/tb_<module>.v`, in dòng chứa `PASS` khi đạt và `[FAIL]` (hoặc `FAILED`, `TIMEOUT`, `ERROR`) khi lỗi: `run_tb.sh` đếm theo các chuỗi này.
- Số liệu báo cáo phải truy được về log: `results/hardware_logs/`, báo cáo Quartus, hoặc lệnh ở trên.
