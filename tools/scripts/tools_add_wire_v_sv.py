"""Thêm `wire` cho cổng module chưa ghi kiểu net (cần khi dùng `default_nettype none).

Chỉ sửa DANH SÁCH CỔNG của module (từ "module" tới ");" đầu tiên). Không đụng tới
input/output của function/task: đối số hàm là biến, ghi "wire" vào sẽ lỗi cú pháp.
`output reg`, `input wire`, `output logic` giữ nguyên.

Dùng:  python tools/scripts/tools_add_wire_v_sv.py rtl/cnn1d/filelist.f
"""
import re
import sys
from pathlib import Path

PORT = re.compile(r'^(\s*)(input|output|inout)(\s+)(?!(?:wire|reg|logic|var)\b)', re.M)
HEADER = re.compile(r'\bmodule\b.*?\)\s*;', re.S)


def rtl_files(filelist: Path):
    for line in filelist.read_text(encoding='utf-8').splitlines():
        line = line.split('//')[0].strip()
        if line:
            yield filelist.parent / line


def fix_file(path: Path) -> int:
    text = path.read_text(encoding='utf-8')
    count = 0

    def fix_header(m):
        nonlocal count
        seg = m.group(0)
        count += len(PORT.findall(seg))
        return PORT.sub(lambda x: x.group(1) + x.group(2) + ' wire' + x.group(3), seg)

    new = HEADER.sub(fix_header, text)
    if count:
        path.write_text(new, encoding='utf-8')
    return count


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    for path in rtl_files(Path(sys.argv[1])):
        n = fix_file(path)
        if n:
            print(f'{n:3d} cong  {path}')


if __name__ == '__main__':
    main()
