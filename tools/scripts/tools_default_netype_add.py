"""Thêm `default_nettype none ở đầu và `default_nettype wire ở cuối mọi file RTL.

Trả về `wire` ở cuối file để không ảnh hưởng các file biên dịch sau (testbench, IP).
Bỏ qua file đã có. Sau khi chạy, chạy tiếp tools_add_wire_v_sv.py rồi biên dịch:
lỗi vlog-2892 còn lại là net ngầm định thật (gõ sai tên / quên khai báo), phải sửa tay.

Dùng:  python tools/scripts/tools_default_netype_add.py rtl/cnn1d/filelist.f
"""
import sys
from pathlib import Path


def rtl_files(filelist: Path):
    for line in filelist.read_text(encoding='utf-8').splitlines():
        line = line.split('//')[0].strip()
        if line:
            yield filelist.parent / line


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    for path in rtl_files(Path(sys.argv[1])):
        text = path.read_text(encoding='utf-8')
        if '`default_nettype none' in text:
            continue
        path.write_text('`default_nettype none\n' + text.rstrip('\n') + '\n`default_nettype wire\n', encoding='utf-8')
        print('da them:', path)


if __name__ == '__main__':
    main()
