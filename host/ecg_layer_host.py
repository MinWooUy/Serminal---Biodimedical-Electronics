"""Host chay TRON LAYER tren FPGA (ecg_layer_top) va so voi golden.

Hai che do:
  python ecg_layer_host.py COM4 [--record rec101_93600 --beat 0 --last-layer 3]
        -> noi UART, chay layer 0..last-layer, so voi golden tung layer
  python ecg_layer_host.py --dry-run trace.hex [--last-layer N]
        -> KHONG can board/pyserial: ghi chuoi goi (lenh + tra loi ky vong) ra file hex,
           de testbench (tb_ecg_layer_core) phat lai o muc goi va kiem chung giao thuc.

Dieu chinh do chi tiet cua log tren man hinh (file log luon ghi day du):
  (mac dinh)  tuong thuat tung buoc: luong dieu khien + luong du lieu + ket qua so sanh
  -v          them 1 dong cho moi goi UART (lenh, kich thuoc, thoi gian khu hoi)
  -vv         them vai byte dau cua payload gui/nhan

Khoanh vung loi:
  --last-layer N     chi chay den layer N
  --first-layer N    nap thang output GOLDEN cua layer N-1 lam input, chay tu layer N
                     (tach rieng layer N, khong bi loi cua layer truoc lan sang)

Layout (khop conv1d_layer_top): input/output "channel-last" X[t][c], co padding;
weight [Cout][K] voi K = j*Cin + i (weights.h luu [Cout][Cin][Kw] nen phai transpose).
"""
import argparse
import math
import re
import struct
import sys
import time
import zlib
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if not (ROOT / "export" / "weights.h").exists():           # chay tu thu muc khac
    ROOT = Path.cwd()

# ---- ma lenh (layer_cmd_controller) ----
CMD_WRITE_X, CMD_WRITE_W, CMD_WRITE_BIAS, CMD_WRITE_SHIFT = 0x08, 0x09, 0x0A, 0x0B
CMD_SWAP, CMD_CONFIG, CMD_RUN, CMD_READ_GAP, CMD_READ_FMAP = 0x0C, 0x0D, 0x0E, 0x0F, 0x10
RSP_ACK, RSP_NACK, RSP_RESULT = 0x80, 0x81, 0x82
CMD_NAMES = {CMD_WRITE_X: "WRITE_X", CMD_WRITE_W: "WRITE_W", CMD_WRITE_BIAS: "WRITE_BIAS",
             CMD_WRITE_SHIFT: "WRITE_SHIFT", CMD_SWAP: "SWAP", CMD_CONFIG: "CONFIG", CMD_RUN: "RUN",
             CMD_READ_GAP: "READ_GAP", CMD_READ_FMAP: "READ_FMAP"}
RSP_NAMES = {RSP_ACK: "ACK", RSP_NACK: "NACK", RSP_RESULT: "RESULT"}
NACK_CODES = {1: "loi khung/CRC (goi bi hong tren duong truyen)",
              2: "tham so sai (do dai, dia chi vuot bo nho, hoac cau hinh khong hop le)",
              3: "sai trang thai (vd RUN khi chua CONFIG)",
              4: "lenh la (FPGA khong biet ma lenh nay - dang nap sai .sof?)"}

CLK_HZ = 50_000_000
ARRAY_ROWS = ARRAY_COLS = 3          # hinh hoc mang PE cua ecg_layer_top

# ---- mo hinh CNN Lite: (Cout, Cin, Kw, pool, gap, pad_next, out_len) ----
LAYERS = [
    dict(cout=16, cin=1,  kw=7, pool=1, gap=0, pad_next=2, out_len=360),
    dict(cout=32, cin=16, kw=5, pool=1, gap=0, pad_next=1, out_len=180),
    dict(cout=64, cin=32, kw=3, pool=1, gap=0, pad_next=1, out_len=90),
    dict(cout=64, cin=64, kw=3, pool=0, gap=1, pad_next=0, out_len=45),
]
POOL_SHAPES = [(16, 180), (32, 90), (64, 45)]            # (C, L) sau pool cua layer 0..2


def c_array(header: str, ctype: str, name: str) -> np.ndarray:
    m = re.search(rf"static\s+const\s+{ctype}\s+{name}\s*\[[^=]+\]\s*=\s*\{{(.*?)\}};", header, re.S)
    if m is None:
        raise RuntimeError(f"Khong thay {name} trong weights.h")
    return np.array([int(v) for v in re.findall(r"-?\d+", m.group(1))], dtype=np.int64)


def load_model():
    header = (ROOT / "export" / "weights.h").read_text(encoding="utf-8")
    model = []
    for n, L in enumerate(LAYERS):
        w = c_array(header, "int8_t", f"CONV{n}_W").reshape(L["cout"], L["cin"], L["kw"])
        w_acc = w.transpose(0, 2, 1).reshape(L["cout"], L["kw"] * L["cin"])    # [o][j][i] -> [Cout][K]
        model.append(dict(L, k=L["kw"] * L["cin"], step=L["cin"], weight=w_acc.reshape(-1),
                          bias=c_array(header, "int32_t", f"CONV{n}_B"),
                          shift=c_array(header, "int8_t", f"CONV{n}_SHIFT")))
    return model


def load_beat(record: str, beat: int):
    gold = ROOT / "export" / "golden"

    def arr(name, dtype, shape):
        return np.fromfile(gold / f"{record}_beat_{name}.bin", dtype=dtype).reshape(shape)[beat]

    z = arr("z_q", np.int8, (-1, 360)).astype(np.int64)
    x0 = np.concatenate([np.zeros(3, np.int64), z, np.zeros(3, np.int64)])    # padding K=7
    expect = []
    for n, (shape, L) in enumerate(zip(POOL_SHAPES, LAYERS[:3])):
        p = arr(f"pool{n}", np.int8, (-1, *shape)).astype(np.int64).T          # (L, C)
        pad = L["pad_next"]
        expect.append(np.pad(p, ((pad, pad), (0, 0))).reshape(-1))
    gap = arr("gap", np.int32, (-1, 64)).astype(np.int64)
    return x0, expect, gap


# ---------------------------------------------------------------- log
class Log:
    """Ghi MOI THU vao file; man hinh chi hien cac dong co level <= verbose."""

    def __init__(self, path=None, verbose=0):
        self.verbose = verbose
        self.path = path
        self.f = open(path, "w", encoding="utf-8") if path else None

    def out(self, msg="", level=0):
        if self.f:
            self.f.write(msg + "\n")
            self.f.flush()
        if level <= self.verbose:
            print(msg, flush=True)

    def close(self):
        if self.f:
            self.f.close()


# ---------------------------------------------------------------- thiet bi
class SerialDevice:
    """Noi UART that, dung Accelerator.exchange cua ecg_host.py."""

    def __init__(self, port, baud, timeout):
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import ecg_host as eh
        self.acc = eh.Accelerator(port, baud, timeout, 3, 3)

    def exchange(self, cmd, payload, rsp, expect=None, cmp_len=None):
        # So sanh du lieu do LayerRunner lam (de bao vi tri sai dau tien), o day chi tra ve payload.
        return bytes(self.acc.exchange(cmd, payload, rsp).payload)

    def close(self):
        self.acc.close()


class DryRunDevice:
    """Chi ghi lai chuoi goi + tra loi ky vong de testbench phat lai."""

    def __init__(self):
        self.packets = []

    def exchange(self, cmd, payload, rsp, expect=None, cmp_len=None):
        if rsp == RSP_ACK:
            expect = bytes((cmd, 0))
            cmp_len = 2
        elif expect is None:
            expect = b""
            cmp_len = 0
        elif cmp_len is None:
            cmp_len = len(expect)
        self.packets.append((cmd, bytes(payload), rsp, bytes(expect), cmp_len))
        return expect

    def write_trace(self, path):
        out = bytearray(struct.pack("<I", len(self.packets)))
        for cmd, payload, rsp, expect, cmp_len in self.packets:
            out += bytes((cmd,)) + struct.pack("<H", len(payload)) + payload
            out += bytes((rsp,)) + struct.pack("<HH", len(expect), cmp_len) + expect
        Path(path).write_text("\n".join(f"{b:02x}" for b in out) + "\n")
        return len(out)

    def close(self):
        pass


class LinkError(RuntimeError):
    """Loi o tang giao thuc/UART (NACK, timeout, CRC...), kem ngu canh de chan doan."""

    def __init__(self, cmd, name, msg, note=""):
        super().__init__(msg)
        self.cmd, self.name, self.msg, self.note = cmd, name, msg, note


class TracedDevice:
    """Boc mot thiet bi: dem goi/byte, do thoi gian tung goi, ghi 1 dong cho moi goi."""

    def __init__(self, dev, log):
        self.dev, self.log = dev, log
        self.packets = self.tx_bytes = self.rx_bytes = 0
        self.t0 = time.monotonic()

    def exchange(self, cmd, payload, rsp, expect=None, cmp_len=None, note=""):
        name = CMD_NAMES.get(cmd, f"0x{cmd:02X}")
        seq = self.packets
        t = time.monotonic()
        head = f"    [{t - self.t0:7.3f}s] #{seq:03d} {name:<10} {len(payload):3d}B{(' ' + note) if note else ''}"
        try:
            res = self.dev.exchange(cmd, payload, rsp, expect=expect, cmp_len=cmp_len)
        except Exception as e:                       # noqa: BLE001 - chuyen thanh LinkError co ngu canh
            self.log.out(f"{head} -> LOI: {e}")
            self.packets += 1
            raise LinkError(cmd, name, str(e), note) from e
        dt = time.monotonic() - t
        self.packets += 1
        self.tx_bytes += len(payload)
        self.rx_bytes += len(res)
        extra = ""
        if self.log.verbose >= 2:
            extra = f"   tx[{payload[:10].hex(' ')}{'..' if len(payload) > 10 else ''}] " \
                    f"rx[{bytes(res[:10]).hex(' ')}{'..' if len(res) > 10 else ''}]"
        self.log.out(f"{head} -> {RSP_NAMES.get(rsp, hex(rsp))} {len(res):3d}B  {dt * 1e3:6.1f} ms{extra}", 1)
        return res

    def close(self):
        self.dev.close()


# ---------------------------------------------------------------- tien ich
def u8(a):
    return bytes(int(v) & 0xFF for v in a)


def stats(a):
    a = np.asarray(a)
    return f"{a.size} phan tu, min={int(a.min())}, max={int(a.max())}, crc32=0x{zlib.crc32(u8(a)) & 0xFFFFFFFF:08x}"


def layer_plan(L):
    """Ke hoach tile cua layer controller: so tile va so chu ky uoc luong (~10*K + 30 chu ky/tile)."""
    rows = 2 if L["pool"] else ARRAY_ROWS                    # co pool: moi tile 2 hang vi tri
    row_tiles = math.ceil(L["out_len"] / rows)
    col_tiles = math.ceil(L["cout"] / ARRAY_COLS)
    tiles = row_tiles * col_tiles
    return rows, row_tiles, col_tiles, tiles, tiles * (10 * L["k"] + 30)


def out_shape(L):
    """(T, C) cua output sau pool + padding cho layer ke tiep."""
    t = L["out_len"] // 2 if L["pool"] else L["out_len"]
    return t + 2 * L["pad_next"], L["cout"]


def decode_link_error(e):
    """Doi thong bao loi tho thanh giai thich + goi y kiem tra."""
    m = re.search(r"original_cmd=(\d+), error=(\d+)", e.msg)
    if m:
        code = int(m.group(2))
        return [f"FPGA tu choi lenh {e.name} (NACK ma {code}): {NACK_CODES.get(code, 'khong ro')}"] + {
            1: ["Goi bi hong: thu lai, kiem tra day UART/baud."],
            2: ["Xem lai tham so lenh vua gui (do dai, dia chi, cau hinh) so voi layer_cmd_controller.v."],
            3: ["Sai thu tu lenh: moi layer phai di WRITE_W -> SWAP -> WRITE_BIAS -> WRITE_SHIFT -> CONFIG -> RUN."],
            4: ["Kiem tra da nap dung file .sof cua ecg_layer_top (khong phai ecg_accel_top cu)."],
        }.get(code, [])
    if "timeout" in e.msg.lower():
        return [f"FPGA khong tra loi lenh {e.name} trong thoi han.",
                "Kiem tra: (1) da nap dung .sof ecg_layer_top chua, (2) dung COM va baud (115200),",
                "          (3) nhan reset tren board roi chay lai, (4) cong UART_RX/TX cua top da noi dung chan chua."]
    if "CRC" in e.msg or "Sequence" in e.msg:
        return [f"Goi tra loi cua lenh {e.name} bi loi khung/CRC/seq: {e.msg}",
                "Thuong do nhieu tren duong truyen hoac FPGA con dang gui goi cu: nhan reset roi chay lai."]
    return [f"Loi giao thuc khi gui {e.name}: {e.msg}"]


# ---------------------------------------------------------------- chay layer
class LayerRunner:
    def __init__(self, dev, log, baud=115200, dump_dir=None):
        self.dev = dev if isinstance(dev, TracedDevice) else TracedDevice(dev, log)
        self.log = log
        self.baud = baud
        self.dump_dir = dump_dir

    # ---- bo cuc log ----
    @contextmanager
    def phase(self, tag, title, detail=()):
        d = self.dev
        self.log.out(f"  [{tag}] {title}")
        for line in detail:
            self.log.out(f"        {line}")
        p0, tx0, rx0, t0 = d.packets, d.tx_bytes, d.rx_bytes, time.monotonic()
        yield
        self.log.out(f"        => {d.packets - p0} goi | gui {d.tx_bytes - tx0} B | nhan {d.rx_bytes - rx0} B "
                     f"| {time.monotonic() - t0:.2f} s")

    def write_bytes(self, cmd, data, chunk=254):
        for off in range(0, len(data), chunk):
            part = data[off:off + chunk]
            self.dev.exchange(cmd, struct.pack("<H", off) + u8(part), RSP_ACK,
                              note=f"addr {off}..{off + len(part) - 1}")

    def write_bias(self, bias, chunk=32):
        for off in range(0, len(bias), chunk):
            part = bias[off:off + chunk]
            self.dev.exchange(CMD_WRITE_BIAS, struct.pack("<H", off) + struct.pack(f"<{len(part)}i", *[int(v) for v in part]),
                              RSP_ACK, note=f"kenh {off}..{off + len(part) - 1}")

    # ---- so sanh ----
    def compare_map(self, name, got, exp, C, pad, rpt):
        """So output layer (T x C, channel-last) voi golden; mo ta cho ro vi tri/kieu sai."""
        if len(got) != len(exp):
            self.log.out(f"        [FAIL] {name}: nhan {len(got)} byte, ky vong {len(exp)} byte (sai kich thuoc)")
            return False
        G = np.frombuffer(bytes(got), np.uint8).view(np.int8).reshape(-1, C).astype(np.int64)
        E = np.frombuffer(bytes(exp), np.uint8).view(np.int8).reshape(-1, C).astype(np.int64)
        T = G.shape[0]
        bad = G != E
        nb = int(bad.sum())
        if nb == 0:
            self.log.out(f"        [PASS] {name}: khop {G.size}/{G.size} byte  ({T} vi tri x {C} kenh; "
                         f"output co {int((E != 0).sum())} gia tri khac 0)")
            return True

        D = G - E
        self.log.out(f"        [FAIL] {name}: SAI {nb}/{G.size} byte ({100.0 * nb / G.size:.1f}%)  ({T} vi tri x {C} kenh)")

        # theo kenh
        per_c = bad.sum(axis=0)
        bad_c = [int(c) for c in np.where(per_c > 0)[0]]
        if len(bad_c) == C:
            self.log.out(f"          theo kenh : MOI kenh deu co loi (min {int(per_c.min())}, max {int(per_c.max())} loi/kenh)")
        else:
            lst = ", ".join(f"c{c}:{int(per_c[c])}" for c in bad_c[:16]) + (" ..." if len(bad_c) > 16 else "")
            self.log.out(f"          theo kenh : {len(bad_c)}/{C} kenh co loi -> {lst}")
            good = [c for c in range(C) if per_c[c] == 0]
            self.log.out(f"                      kenh dung hoan toan: {good if len(good) <= 16 else f'{len(good)} kenh'}")

        # theo vi tri t
        ts = np.where(bad.any(axis=1))[0]
        in_pad = int(bad[[t for t in range(T) if t < pad or t >= T - pad]].sum()) if pad else 0
        self.log.out(f"          theo t    : {len(ts)}/{T} vi tri co loi, dau tien t={int(ts[0])}, cuoi cung t={int(ts[-1])}"
                     f"{f', trong do {in_pad} loi nam o vung padding' if pad else ''}")

        # bang mau
        first = np.argwhere(bad)
        self.log.out("          mau loi dau tien (t, c, nhan, ky vong, lech):")
        for t, c in first[:6]:
            self.log.out(f"            t={int(t):4d} c={int(c):3d}   got={int(G[t, c]):5d}  exp={int(E[t, c]):5d}  diff={int(D[t, c]):+d}")

        # mot hang day du de nhin theo kenh
        t0 = int(first[0][0])
        w = min(C, 16)
        self.log.out(f"          hang t={t0} (kenh 0..{w - 1}):")
        self.log.out("            got : " + " ".join(f"{int(v):4d}" for v in G[t0, :w]))
        self.log.out("            exp : " + " ".join(f"{int(v):4d}" for v in E[t0, :w]))
        self.log.out("            sai : " + " ".join("   ^" if b else "    " for b in bad[t0, :w]))

        # goi y nguyen nhan
        hints = []
        if not G.any():
            hints.append("Output TOAN 0: fmap_writer khong ghi, hoac doc nham bank (chua swap), hoac RUN khong chay that.")
        else:
            for s in (1, -1, 2, -2):
                lo, hi = (s, T) if s > 0 else (0, T + s)
                if (G[lo:hi] == np.roll(E, s, axis=0)[lo:hi]).mean() > 0.9:
                    hints.append(f"Output GIONG ky vong khi dich {s:+d} vi tri theo t: sai dia chi/thu tu ghi (out_pos, padding, pool).")
            if C > 1:
                for s in (1, -1):
                    if (G == np.roll(E, s, axis=1)).mean() > 0.9:
                        hints.append(f"Output GIONG ky vong khi dich {s:+d} kenh: sai chi so cot/kenh khi ghi.")
        if len(bad_c) < C and len(bad_c) > 0:
            mods = {c % ARRAY_COLS for c in bad_c}
            if len(mods) == 1:
                hints.append(f"Moi loi o kenh c % {ARRAY_COLS} == {mods.pop()}: nghi MOT COT PE (weight/bias/shift cua cot do).")
        inner = [int(t) - pad for t in ts if pad <= t < T - pad]
        if rpt > 1 and len(inner) > 1 and len({v % rpt for v in inner}) == 1:
            hints.append(f"Moi loi o hang tile {inner[0] % rpt}/{rpt}: nghi MOT HANG PE (hoac nhanh max cua cap pool).")
        if int(np.abs(D[bad]).max()) <= 1:
            hints.append("Chi lech +-1: nghi lam tron (rshift_round), shift sai 1, hoac bias.")
        if (G[bad] == 0).mean() > 0.8:
            hints.append("Phan lon byte sai co got=0 (ky vong khac 0): ReLU/clamp qua tay, hoac writer bo ghi vi tri do.")
        if pad and in_pad == nb:
            hints.append("Chi sai o vung padding: so hang padding (row_pad) hoac zero-fill chua dung.")
        if hints:
            self.log.out("          goi y     :")
            for h in hints:
                self.log.out(f"            - {h}")

        if self.dump_dir:
            p = Path(self.dump_dir) / f"{name.replace(' ', '_')}_got_exp.npz"
            np.savez(p, got=G, exp=E)
            self.log.out(f"          du lieu got/exp da luu: {p}")
        return False

    def compare_gap(self, got, gap):
        exp = np.array([int(v) & 0xFFFF for v in gap], dtype=np.int64)
        G = np.array(struct.unpack(f"<{len(got) // 2}H", bytes(got)), dtype=np.int64)
        if len(G) != len(exp):
            self.log.out(f"        [FAIL] GAP: nhan {len(G)} kenh, ky vong {len(exp)}")
            return False
        bad = np.where(G != exp)[0]
        if len(bad) == 0:
            self.log.out(f"        [PASS] GAP: khop {len(exp)}/{len(exp)} kenh (gia tri {int(exp.min())}..{int(exp.max())})")
            return True
        D = G - exp
        self.log.out(f"        [FAIL] GAP: SAI {len(bad)}/{len(exp)} kenh")
        for c in bad[:8]:
            self.log.out(f"            c={int(c):3d}   got={int(G[c]):6d}  exp={int(exp[c]):6d}  diff={int(D[c]):+d}")
        hints = []
        if not G.any():
            hints.append("GAP toan 0: writer che do GAP khong cong don, hoac layer 3 khong chay.")
        if len(set(D[bad].tolist())) == 1:
            hints.append(f"Moi kenh sai deu lech cung mot hang so ({int(D[bad[0]]):+d}): nghi bias hoac thieu/thua mot vi tri.")
        if (np.abs(D[bad]) <= np.maximum(1, exp[bad] // 100)).all():
            hints.append("Lech nho (<1%): sai o vai gia tri dau vao (layer 2 -> layer 3), khong phai o buoc cong don.")
        if hints:
            self.log.out("          goi y     :")
            for h in hints:
                self.log.out(f"            - {h}")
        return False

    # ---- mot layer ----
    def run_layer(self, n, L, x, expect, gap, first):
        d = self.dev
        rows, row_tiles, col_tiles, tiles, est = layer_plan(L)
        T_out, C_out = out_shape(L)
        xbytes = len(x) if first else 0
        up = xbytes + len(L["weight"]) + 4 * len(L["bias"]) + len(L["shift"])
        self.log.out("")
        self.log.out("=" * 78)
        self.log.out(f"LAYER {n}  conv{n}: Cin={L['cin']} -> Cout={L['cout']}, Kw={L['kw']} (K={L['k']}), "
                     f"out_len={L['out_len']}, pool={L['pool']}, gap={L['gap']}, pad_next={L['pad_next']}")
        self.log.out(f"  Ke hoach : mang {ARRAY_ROWS}x{ARRAY_COLS}, {rows} hang vi tri/tile{' (do co pool)' if L['pool'] else ''}"
                     f" -> {row_tiles} x {col_tiles} = {tiles} tile, uoc ~{est:,} chu ky (~{est / CLK_HZ * 1e3:.1f} ms @50MHz)")
        self.log.out(f"  Truyen   : ~{up:,} byte tu host, uoc ~{up * 10 / self.baud * 1.15:.1f} s o {self.baud} baud")
        self.log.out(f"  Du lieu  : {'output GOLDEN cua layer ' + str(n - 1) if (first and n) else ('x0 (z_q + pad)' if first else 'output layer truoc (da nam trong SRAM)')}"
                     f"  ->  [PE 3x3 -> post -> {'maxpool -> ' if L['pool'] else ''}fmap_writer]  ->  "
                     f"{'GAP 64 kenh (16 bit)' if L['gap'] else f'{T_out} vi tri x {C_out} kenh = {T_out * C_out} byte (gom pad)'}")
        self.log.out("-" * 78)

        try:
            # [A] input
            if first:
                cin = L["cin"]
                with self.phase("A", f"Nap input X vao SRAM ifmap (bank GHI), roi SWAP sang bank DOC",
                                [f"nguon: {'x0 = z_q (360) + 3 zero moi dau (pad cho K=7)' if n == 0 else f'golden output layer {n - 1} (da gom padding)'}",
                                 f"hinh dang: {len(x) // cin} vi tri x {cin} kenh = {len(x)} byte | {stats(x)}"]):
                    self.write_bytes(CMD_WRITE_X, x)
                    d.exchange(CMD_SWAP, bytes((1,)), RSP_ACK, note="x_swap (bit0)")
            else:
                self.log.out(f"  [A] Input: lay tu output layer {n - 1}, da nam trong SRAM ifmap "
                             f"(controller tu swap khi layer truoc xong) - khong truyen UART")

            # [B] weight
            with self.phase("B", f"Nap weight vao SRAM weight (bank GHI), roi SWAP",
                            [f"hinh dang: [Cout={L['cout']}][K={L['k']}] voi K = j*Cin + i = {len(L['weight'])} byte",
                             stats(L["weight"]) + f" | 8 byte dau (filter 0): {[int(v) for v in L['weight'][:8]]}"]):
                self.write_bytes(CMD_WRITE_W, L["weight"])
                d.exchange(CMD_SWAP, bytes((2,)), RSP_ACK, note="w_swap (bit1)")

            # [C] bias + shift
            sh = [int(v) for v in L["shift"]]
            with self.phase("C", "Nap bias (INT32) va shift (luong tu hoa) theo kenh",
                            [f"bias : {stats(L['bias'])}",
                             f"shift: {len(sh)} kenh, min={min(sh)}, max={max(sh)}  -> {sh if len(sh) <= 16 else sh[:16] + ['...']}"]):
                self.write_bias(L["bias"])
                self.write_bytes(CMD_WRITE_SHIFT, L["shift"])

            # [D] config
            cfg = struct.pack("<HHHHBBB", L["out_len"], L["cout"], L["k"], L["step"], L["pool"], L["gap"], L["pad_next"])
            with self.phase("D", "CONFIG layer",
                            [f"out_len={L['out_len']} cout={L['cout']} k_len={L['k']} window_step={L['step']} "
                             f"(= Cin: buoc truot cua cua so) pool={L['pool']} gap={L['gap']} pad_next={L['pad_next']}"]):
                d.exchange(CMD_CONFIG, cfg, RSP_ACK, note="(cau hinh)")

            # [E] run
            with self.phase("E", "RUN: FPGA tu lap tile (PE 3x3) -> post -> pool -> ghi fmap -> swap, host chi doi ket qua"):
                res = d.exchange(CMD_RUN, b"", RSP_RESULT, expect=bytes(5), cmp_len=1, note="doi layer chay xong")
                status = res[0]
                cycles = struct.unpack("<I", res[1:5])[0] if len(res) == 5 else 0
                dev_est = f"; uoc luong {est:,} ({100.0 * (cycles - est) / est:+.1f}%)" if cycles else ""
                self.log.out(f"        trang thai={status} ({'OK' if status == 0 else 'writer OVERRUN'}), "
                             f"{cycles:,} chu ky = {cycles / CLK_HZ * 1e3:.2f} ms @50MHz{dev_est}")
                if status != 0:
                    self.log.out("        writer overrun: bo ghi fmap bi tran (tile sau bat dau khi tile truoc chua ghi xong).")

            # [F] read + compare
            ok = status == 0
            if n < 3:
                exp = u8(expect[n])
                with self.phase("F", f"Doc lai output layer ({T_out} x {C_out}) bang READ_FMAP va so voi golden",
                                [f"{len(exp)} byte, doc theo goi 250 byte"]):
                    got_all = bytearray()
                    for off in range(0, len(exp), 250):
                        ln = min(250, len(exp) - off)
                        got_all += self.dev.exchange(CMD_READ_FMAP, struct.pack("<HB", off, ln), RSP_RESULT,
                                                     expect=exp[off:off + ln], note=f"off {off}..{off + ln - 1}")
                    ok = self.compare_map(f"layer{n}_fmap", bytes(got_all), exp, C_out, L["pad_next"], rows // (2 if L["pool"] else 1)) and ok
            else:
                exp = b"".join(struct.pack("<H", int(v) & 0xFFFF) for v in gap)
                with self.phase("F", "Doc GAP 64 kenh (u16) bang READ_GAP va so voi golden"):
                    got = self.dev.exchange(CMD_READ_GAP, b"", RSP_RESULT, expect=exp, note="128 byte")
                    ok = self.compare_gap(got, gap) and ok
        except LinkError as e:
            self.log.out("")
            self.log.out(f"  [LOI GIAO THUC] tai lenh {e.name} {e.note}")
            for line in decode_link_error(e):
                self.log.out(f"    {line}")
            return False, 0
        self.log.out(f"LAYER {n}: {'PASS' if ok else 'FAIL'}")
        return ok, cycles

    def run(self, model, x0, expect, gap, last_layer=3, first_layer=0, keep_going=False):
        self.log.out("Luong dieu khien moi layer : WRITE_W -> SWAP -> WRITE_BIAS -> WRITE_SHIFT -> CONFIG -> RUN -> READ")
        self.log.out("Luong du lieu moi layer    : host --UART--> SRAM weight(ghi) -swap-> SRAM weight(doc) -> feeder -> PE 3x3")
        self.log.out("                             SRAM ifmap(doc) -> feeder ->  PE 3x3 -> post(+bias >> shift, ReLU, clamp +-127)")
        self.log.out("                             -> [maxpool 2] -> fmap_writer -> SRAM ifmap(ghi, them pad) -swap-> input layer sau")
        rows = []
        t_all = time.monotonic()
        for n in range(first_layer, last_layer + 1):
            first = n == first_layer
            x = (x0 if n == 0 else expect[n - 1]) if first else None
            ok, cycles = self.run_layer(n, model[n], x, expect, gap, first)
            rows.append((n, ok, cycles))
            if not ok and not keep_going and n < last_layer:
                self.log.out("")
                self.log.out(f"DUNG o layer {n}: cac layer sau chay tren du lieu sai nen khong co y nghia. "
                             f"Dung --first-layer {n} de nap golden vao va tach rieng layer {n}, "
                             f"hoac --keep-going de chay tiep.")
                break

        self.log.out("")
        self.log.out("=" * 78)
        self.log.out("TONG KET")
        self.log.out(f"  {'layer':<6}{'ket qua':<10}{'chu ky':>14}{'thoi gian @50MHz':>20}")
        total = 0
        for n, ok, cycles in rows:
            total += cycles
            self.log.out(f"  {n:<6}{'PASS' if ok else 'FAIL':<10}{cycles:>14,}{cycles / CLK_HZ * 1e3:>17.2f} ms")
        self.log.out(f"  {'tong':<16}{total:>14,}{total / CLK_HZ * 1e3:>17.2f} ms   (phan cung)")
        self.log.out(f"  tong thoi gian chay thuc: {time.monotonic() - t_all:.1f} s  (chu yeu la truyen UART), "
                     f"{self.dev.packets} goi, {self.dev.tx_bytes:,} B gui, {self.dev.rx_bytes:,} B nhan")
        return all(ok for _, ok, _ in rows) and len(rows) == last_layer - first_layer + 1


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("port", nargs="?", help="Cong COM, vi du COM4")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--timeout", type=float, default=5.0)
    ap.add_argument("--root", help="Goc repo (chua export/weights.h); mac dinh: thu muc cha cua host/")
    ap.add_argument("--record", default="rec101_93600")
    ap.add_argument("--beat", type=int, default=0)
    ap.add_argument("--first-layer", type=int, default=0, choices=range(4),
                    help="Nap golden output layer N-1 lam input va chay tu layer N (tach rieng 1 layer)")
    ap.add_argument("--last-layer", type=int, default=3, choices=range(4))
    ap.add_argument("--keep-going", action="store_true", help="Chay tiep cac layer sau du layer truoc sai")
    ap.add_argument("-v", "--verbose", action="count", default=0, help="-v: moi goi 1 dong; -vv: them byte dau payload")
    ap.add_argument("--log-dir", help="Thu muc ghi log (mac dinh: results/hardware_logs)")
    ap.add_argument("--no-log", action="store_true", help="Khong ghi file log")
    ap.add_argument("--dry-run", metavar="TRACE", help="Ghi chuoi goi ra file hex cho testbench (khong can board)")
    args = ap.parse_args()
    if args.root:
        global ROOT
        ROOT = Path(args.root)
    if args.first_layer > args.last_layer:
        ap.error("--first-layer phai <= --last-layer")

    model = load_model()
    x0, expect, gap = load_beat(args.record, args.beat)
    if args.dry_run:
        dev = DryRunDevice()
        log = Log(None, verbose=-1)
        LayerRunner(dev, log).run(model, x0, expect, gap, args.last_layer, args.first_layer)
        size = dev.write_trace(args.dry_run)
        print(f"{len(dev.packets)} goi, {size} byte -> {args.dry_run}")
        return 0
    if not args.port:
        ap.error("Can cong COM hoac --dry-run")

    log_path = dump_dir = None
    if not args.no_log:
        dump_dir = Path(args.log_dir) if args.log_dir else ROOT / "results" / "hardware_logs"
        dump_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_path = dump_dir / f"layer{args.first_layer}-{args.last_layer}_{args.record}_beat{args.beat:03d}_{stamp}.log"
    log = Log(log_path, verbose=args.verbose)
    log.out(f"ECG layer host | cong {args.port} @ {args.baud} baud | record={args.record} beat={args.beat} | "
            f"layer {args.first_layer}..{args.last_layer} | {datetime.now():%Y-%m-%d %H:%M:%S}")
    if log_path:
        log.out(f"Log day du: {log_path}")
    try:
        dev = SerialDevice(args.port, args.baud, args.timeout)
    except Exception as e:                           # noqa: BLE001
        log.out(f"Khong mo duoc {args.port}: {e}")
        log.out("Kiem tra: dung ten cong COM (Device Manager), cong khong bi chuong trinh khac giu, da cam cap USB.")
        log.close()
        return 2
    try:
        ok = LayerRunner(dev, log, args.baud, dump_dir).run(model, x0, expect, gap, args.last_layer,
                                                            args.first_layer, args.keep_going)
    finally:
        dev.close()
    log.out("")
    log.out("KET QUA: PASS" if ok else "KET QUA: FAIL")
    log.close()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
