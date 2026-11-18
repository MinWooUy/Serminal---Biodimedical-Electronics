# Loi da biet va viec can tich hop

File nay ghi lai cac van de tim thay khi them pipeline cau hinh va du lieu.
No tach rieng khoi README giao thuc de nguoi sau co the dong tung muc ma
khong phai sua lai mo ta thiet ke.

## Dang mo

1. `cnn_pipeline_controller.v` va `layer_config_ram.v` hien la cac khoi dieu
   phoi doc lap. Chung da duoc them vao `filelist.f` va co testbench rieng,
   nhung chua noi vao `layer_core.v` hay bo giai ma lenh UART. Cac ma CMD da
   danh rieng cho buoc tiep theo: `0x11 CONFIG_ALL`, `0x12 LOAD_COMMIT`,
   `0x13 RUN_ALL`, `0x14 GET_RESULT`; can viet payload decoder va noi FSM.
2. Duong UART/controller hien van la stop-and-wait: moi request duoc ACK rieng
   va chua co FIFO request. Khong gui burst frame truoc khi FSM nhan va co che
   commit/xu ly loi duoc cai dat va kiem thu.
3. Kha nang giu toan bo weight van bi gioi han boi `WGT_MEM_DEPTH=16384`.
   Bon tensor Conv can tong cong 21.104 byte. Hoac nap weight theo tung layer,
   hoac them weight memory lon/rieng truoc khi ket luan da nap model mot lan.
4. Cac thay doi UART hien dat baud mac dinh 921600 trong top-level va host.
   Chua kiem tra van hanh UART 921600 tren board that; can tach rieng viec
   kiem thu baud khoi viec kiem thu tinh dung dan cua pipeline.

## Tuong thich testbench

`tb_uart_crc_top.v` con goi interface tham so cu cua `ecg_accel_top`
(`MEM_DEPTH` va `POST_ENABLE`). Top-level hien tai dung
`IFMAP_MEM_DEPTH`, `WGT_MEM_DEPTH`, `COUT_MAX`, va duong layer dung
`layer_core`. Can chuyen testbench nay truoc khi xem no la bai test UART
end-to-end hop le.

## Da duoc test tap trung

- Bon muc cau hinh layer duoc luu doc lap.
- Lenh chay bi tu choi khi chua du bon muc cau hinh.
- Moi layer nhan dung cau hinh da luu cua minh.
- Huong bank feature-map doi sau moi layer hoan tat.
- Trang thai cuoi chi dat duoc sau bon su kien `layer_done`.