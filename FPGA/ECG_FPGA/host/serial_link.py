"""Serial port side of the protocol v2 host: transport, raw byte log, loopback test, ping diagnostics.

Standard library + pyserial only (shipped in the FPGA hand-off package next to proto_v2.py and host.py).
Messages for the person at the bench are in Vietnamese.
"""

import time

try:
    from python.link import proto_v2 as pv2
except ImportError:                       # stand-alone copy in the FPGA hand-off package
    import proto_v2 as pv2


def utf8_console() -> None:
    """Windows consoles default to a legacy code page that cannot print Vietnamese: switch stdout/stderr to UTF-8."""
    import sys
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


class SerialTransport:
    """write(bytes), read(timeout) -> bytes, clock() on a serial port, 115200 8N1 by default.
    `port` may also be a pyserial URL, e.g. 'loop://' (software loopback, no hardware)."""

    def __init__(self, port: str, baud: int = 115200):
        import serial                                   # pyserial
        if "://" in port:
            self.ser = serial.serial_for_url(port, baudrate=baud, bytesize=8, parity="N", stopbits=1, timeout=0)
        else:
            self.ser = serial.Serial(port, baud, bytesize=8, parity="N", stopbits=1, timeout=0)
        self.port, self.baud = port, baud
        self.ser.reset_input_buffer()

    def write(self, data: bytes) -> None:
        self.ser.write(data)
        self.ser.flush()

    def read(self, timeout: float) -> bytes:
        end = time.perf_counter() + timeout
        while True:
            n = self.ser.in_waiting
            if n:
                return self.ser.read(n)
            if time.perf_counter() >= end:
                return b""
            time.sleep(0.001)

    def clock(self) -> float:
        return time.perf_counter()

    def close(self) -> None:
        self.ser.close()


class RawLog:
    """Wraps a transport and writes every byte sent / received with a time stamp (s since start):
        <time> TX|RX <n bytes> <hex bytes>"""

    def __init__(self, transport, path, header: str = ""):
        self.t = transport
        self.f = open(path, "w", encoding="ascii", errors="replace")
        self.t0 = transport.clock()
        self.f.write(f"# raw byte log, protocol v2; {time.strftime('%Y-%m-%d %H:%M:%S')}; {header}\n"
                     "# time_s dir n hex\n")

    def _log(self, d: str, data: bytes) -> None:
        self.f.write(f"{self.t.clock() - self.t0:12.6f} {d} {len(data):5d} {data.hex(' ')}\n")
        self.f.flush()

    def write(self, data: bytes) -> None:
        self._log("TX", data)
        self.t.write(data)

    def read(self, timeout: float) -> bytes:
        data = self.t.read(timeout)
        if data:
            self._log("RX", data)
        return data

    def clock(self) -> float:
        return self.t.clock()

    def close(self) -> None:
        self.f.close()
        if hasattr(self.t, "close"):
            self.t.close()


# ---------------------------------------------------------------- loopback (TX shorted to RX, no FPGA)

def loopback(transport, count: int = 3, timeout: float = 0.5) -> dict:
    """Send `count` PING frames; with TX shorted to RX every byte must come back unchanged."""
    frame = pv2.encode(pv2.Ping())
    sent, got = b"", b""
    for _ in range(count):
        transport.write(frame)
        sent += frame
        end = transport.clock() + timeout
        while len(got) < len(sent) and transport.clock() < end:
            got += transport.read(end - transport.clock())
    end = transport.clock() + 0.05                     # anything extra arriving late
    while transport.clock() < end:
        got += transport.read(end - transport.clock())
    ok = got == sent
    if ok:
        msg = [f"LOOPBACK ĐẠT: gửi {len(sent)} byte, nhận lại đúng {len(got)} byte. Driver, cổng COM và dây đều ổn;",
               "tháo dây nối tắt TX–RX rồi nối module với FPGA (TX module → RX FPGA, RX module → TX FPGA, GND chung)."]
    elif not got:
        msg = [f"LOOPBACK TRƯỢT: gửi {len(sent)} byte, KHÔNG nhận lại byte nào.",
               "- Dây nối tắt có thật sự nối chân TX với chân RX của module USB-UART không?",
               "- Có đúng cổng COM không (Device Manager → Ports (COM & LPT); rút cắm module để thấy cổng nào mất/hiện)?",
               "- Driver của module (CH340 / CP210x / FTDI) đã cài chưa?"]
    else:
        msg = [f"LOOPBACK TRƯỢT: gửi {len(sent)} byte, nhận {len(got)} byte KHÁC dữ liệu đã gửi.",
               f"- gửi: {sent[:21].hex(' ')}", f"- nhận: {got[:21].hex(' ')}",
               "- Dây lỏng / tiếp xúc kém, hoặc còn thiết bị khác đang nối vào chân RX (đang nối FPGA?)."]
    return {"ok": ok, "sent": len(sent), "received": len(got), "messages": msg}


# ---------------------------------------------------------------- ping diagnostics

def check_pong(pong, expected_crc: int) -> list[str]:
    bad = []
    if pong.proto_ver != pv2.PROTO_VER:
        bad.append(f"proto_ver {pong.proto_ver} != {pv2.PROTO_VER}")
    if pong.mode != pv2.MODE_CNN_ONLY:
        bad.append(f"mode {pong.mode} != {pv2.MODE_CNN_ONLY}")
    if pong.weights_crc32 != expected_crc:
        bad.append(f"weights_crc32 0x{pong.weights_crc32:08X} != frozen model 0x{expected_crc:08X}")
    return bad


def describe_pong(p) -> str:
    return (f"PONG: proto_ver {p.proto_ver}, mode {p.mode}, build_id 0x{p.build_id:08X}, "
            f"weights_crc32 0x{p.weights_crc32:08X}")


def diagnose_ping(host, expected_crc: int, pong=None) -> list[str]:
    """Plain-language diagnosis (Vietnamese) of a failed PING (pong=None) or of a PONG that does not match."""
    if pong is not None:
        out = []
        if pong.weights_crc32 != expected_crc:
            out += [f"PONG báo weights_crc32 0x{pong.weights_crc32:08X}, nhưng mô hình đóng băng là 0x{expected_crc:08X}:",
                    "FPGA đang dùng BỘ TRỌNG SỐ KHÁC (nạp nhầm file mem/, bitstream cũ, hoặc WEIGHTS_CRC32 trong",
                    "ecg_params.vh không lấy từ gói bàn giao này). Nạp lại mem/ và ecg_params.vh của gói này rồi build lại."]
        if pong.proto_ver != pv2.PROTO_VER:
            out.append(f"FPGA dùng phiên bản giao thức {pong.proto_ver}, PC dùng {pv2.PROTO_VER}: bitstream không đúng đặc tả v2.")
        if pong.mode != pv2.MODE_CNN_ONLY:
            out.append(f"mode = {pong.mode}, phải là {pv2.MODE_CNN_ONLY} (chỉ CNN).")
        return out
    ping = pv2.encode(pv2.Ping())
    d, s = host.dec, host.stats
    head = bytes(host.rx_head)
    if host.rx_bytes == 0:
        return [f"Không nhận được byte nào sau {s.sent} lần gửi PING. Kiểm tra lần lượt:",
                "- Dây TX/RX có bị ĐẢO không: TX của module USB-UART phải nối vào RX của FPGA và ngược lại.",
                "- Có nối GND chung giữa module và board không.",
                "- Đúng cổng COM chưa (chạy lệnh loopback trước để chắc cổng và dây tốt).",
                "- FPGA đã nạp bitstream và đang chạy chưa (nút reset đã nhả, chân UART gán đúng trong Quartus)."]
    if head and head == (ping * (len(head) // len(ping) + 1))[:len(head)]:
        return [f"Nhận lại ĐÚNG các byte PING vừa gửi ({host.rx_bytes} byte): TX đang nối thẳng sang RX.",
                "- Dây nối tắt của bước loopback chưa tháo, hoặc FPGA đang phản xạ nguyên dữ liệu (chưa chạy đúng thiết kế)."]
    if d.frames_ok == 0:
        return [f"Có {host.rx_bytes} byte về nhưng không ghép được khung hợp lệ nào "
                f"(lỗi CRC {d.bad_crc}, LEN {d.bad_len}, TYPE {d.bad_type}; bỏ qua {d.skipped_bytes} byte).",
                "- Sai tốc độ baud: cả hai phía phải là 115200, 8N1.",
                "- Mức điện áp: module USB-UART phải đặt 3.3 V (KHÔNG dùng 5 V với chân FPGA).",
                "- Dây lỏng / nhiễu; FPGA phải dùng khung v2 (A5 5A, TYPE, LEN 2 byte little-endian, CRC-16 little-endian).",
                f"- Những byte đầu nhận được: {head[:32].hex(' ')}"]
    nacks = {k: v for k, v in s.nacks.items() if v}
    if nacks:
        names = {1: "CRC sai", 2: "LEN sai với TYPE", 3: "TYPE lạ", 4: "đang bận"}
        return [f"FPGA trả NACK thay vì PONG: " + ", ".join(f"{names.get(k, k)} × {v}" for k, v in nacks.items()) + ".",
                "- CRC sai: nhiễu / sai baud trên chiều PC → FPGA, hoặc FPGA tính CRC khác đặc tả (CRC-16/CCITT-FALSE,",
                "  trên TYPE + LEN + PAYLOAD). Kiểm: khung PING phải là A5 5A 01 00 00 AC FB.",
                "- LEN sai / TYPE lạ: FPGA đọc LEN 1 byte thay vì 2 byte little-endian, hoặc chưa hỗ trợ PING (0x01)."]
    return [f"Nhận {d.frames_ok} khung hợp lệ nhưng không có PONG ({s.ignored} gói khác). FPGA phải trả PONG (0x02) "
            "cho mỗi PING."]
