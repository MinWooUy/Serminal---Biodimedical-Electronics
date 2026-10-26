# Post processing cho Conv1D INT8

`conv1d_post_process.v` xử lý một tile có hàng = vị trí thời gian,
cột = kênh đầu ra. Mỗi cột có bias INT32 và shift signed INT8 riêng.
Thứ tự đóng gói: phần tử `(row*COLS+col)` nằm ở các bit thấp trước;
bias/shift của cột 0 cũng nằm ở các bit thấp nhất.

Với từng phần tử đang hoạt động:

```
a = signed(acc) + signed(bias[col])
s = k[col] + a_in - a_out
q = (a + 2^(s-1)) >>> s   nếu s > 0
q = a                    nếu s = 0
q = a << (-s)            nếu s < 0
out = clamp(q, 0, 127)    nếu ReLU bật
out = clamp(q, -127, 127) nếu ReLU tắt
```

Đúng quy tắc làm tròn của `python/quant/int8.py`, kể cả số âm (half toward
positive infinity). Phép cộng bias và rounding dùng trung gian 64 bit để không
tràn INT32. Shift rất lớn được xử lý theo kết quả bão hòa toán học, không wrap.
Model hợp lệ phải bảo đảm tổng accumulator + bias nằm trong INT32 như golden.
Bias đã chứa BatchNorm được fold; không cần khối BatchNorm riêng.
Các hàng/cột không hoạt động luôn bằng 0, kể cả khi bias khác 0.

Giao diện không có backpressure: tại mỗi cạnh lên có `i_valid=1`, lấy cả tile
và cấu hình; sau cạnh đó `o_tile` và `o_valid` là kết quả đã đăng ký.
Nhận được một tile mỗi chu kỳ. Khi không valid, giữ data và hạ valid.
Consumer phải nhận mọi chu kỳ valid hoặc đặt FIFO phía sau.
Reset bất đồng bộ active-high xóa dữ liệu và valid.

## Tích hợp với conv1d_tile_top

Đặt parameter `POST_ENABLE=1` (mặc định 0), `ACC_W=32`, nối:

- `i_post_shift`: `ARRAY_COLS` shift signed 8 bit, lấy từ model.
- `i_post_relu`: bật cho các block Conv1D.
- `o_post_tile`: tile INT8 theo cùng thứ tự với `o_tile`.
- `o_post_valid`: xung một chu kỳ cho mỗi tile hoàn tất.

Cấu hình post được chốt khi tile nhận `i_start` ở IDLE, cùng với cấu hình
Conv1D. Bộ chặn kết quả lặp chuyển `o_tile_valid` giữ mức của core thành một
lần xử lý duy nhất. Dùng `o_post_valid` để nhận đầu ra INT8; không dùng
`o_done` hoặc `o_tile_valid` thay cho nó. `o_tile`, `o_done` và giao diện INT32
cũ giữ nguyên. Khi POST_ENABLE=0, hai đầu ra post bằng 0; có thể bỏ nối đầu vào.

Bias Conv1D được nạp vào accumulator qua preload; post block chỉ thực hiện
round/shift, ReLU và saturate. `ecg_accel_top` dùng `POST_ENABLE` để chọn chế độ
raw hoặc raw + INT8 post result; `CONFIG_POST` (`0x07`) nạp shift và ReLU qua UART.
của tile; muốn bật qua host cần mở rộng command controller và host protocol.
Chỉ áp dụng post sau khi cộng đủ mọi partial sum của một output; không cộng
bias hay requantize riêng từng lát K rồi cộng các kết quả INT8.

## MaxPool(2)

`maxpool1d2.v` là khối stream riêng, kernel=2, stride=2, so sánh signed INT8.
Mỗi input là một vị trí với `CHANNELS` kênh. Cấp các vị trí thời gian liên tiếp
của cùng nhóm kênh; serialize các hàng tile theo thứ tự thời gian trước khi
đưa vào. Khối giữ vị trí lẻ qua bubble và qua ranh giới tile, nên cặp gồm hàng
cuối tile trước và hàng đầu tile sau vẫn được ghép đúng.

`i_clear` ưu tiên hơn `i_valid`: bỏ mẫu đang chờ khi bắt đầu chuỗi/nhóm kênh
mới. Đuôi lẻ bị bỏ, đúng floor-length pooling. Không clear giữa hai tile liên
tiếp của cùng chuỗi. Đầu ra valid sau mỗi input thứ hai. Không có backpressure.
Khối này chưa tự nối vào bus tile song song; scheduler phải tuần tự hóa các
hàng và điều khiển clear. Các block Conv1D 1–3 của model cần pooling, block 4
không cần. GAP, ghép RR, FC1 và argmax FC2 không thuộc module này; đặc biệt
không đưa logits FC2 qua ReLU/kẹp INT8.

## Kiểm tra

Từ `rtl/cnn1d` chạy:

```
bash sim/run_post_tests.sh
bash sim/run_comm_tests.sh
```

Script post tạo 1924 vector biên + ngẫu nhiên với golden Python số nguyên,
kiểm tra 9 lane, rounding âm, các shift -128..127, saturation, masking, valid,
reset, pooling qua bubble và clear. Test tile kiểm tra thêm 4 kích thước tile,
một kết quả post mỗi tile và giá trị so với kết quả MAC. Có thể đặt IVERILOG,
VVP, IVL_BASE nếu dùng Icarus giải nén ngoài hệ thống.
