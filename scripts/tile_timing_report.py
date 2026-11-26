"""Phan tich log cua rtl/cnn1d/tb/tb_tile_timing.v: tach thoi gian moi o thanh cac pha va do delta.

    rtl/cnn1d/sim/run_tb.sh --param LAST_LAYER=3 tb_tile_timing
    python scripts/tile_timing_report.py rtl/cnn1d/sim/out/tb_tile_timing.log

Moi dong TILE trong log co 14 moc chu ky (xem tb_tile_timing.v).  Cac pha lien tiep, tong bang chu ky tu
o nay den o ke tiep (tau):
    load      t_start  -> t_loaded   hai loader nap xong (tile FSM: S_LOAD_START, S_LOAD_WAIT)
    launch    t_loaded -> t_cload    tile FSM S_CORE_START cap xung start cho loi
    preload   t_cload  -> t_ccomp    Core FSM S_LOAD, S_PRE_KICK, S_PRE_WAIT: bom bias vao chuoi shadow
    compute   t_ccomp  -> t_cdrain   mang PE chay (den khi moi PE nhan last)
    drain     t_cdrain -> t_post     xa chuoi shadow qua PSM (den khi o_tile_valid len)
    post      t_post   -> t_postv    post-process tra ket qua
    wgap      t_postv  -> t_wstart   cho fmap_writer bat dau
    write     t_wstart -> t_wdone    fmap_writer ghi ket qua vao SRAM
    commit    t_wdone  -> t_commit   cho tile_commit (tile_top ve IDLE)
    next      t_commit -> t_start'   layer controller phat o ke tiep
delta = tau - K * max(ru, cu) - (K + ru + cu); delta0 = delta - n_wr.
"""
import re
import sys
from collections import defaultdict

PHASES = ["load", "launch", "preload", "compute", "drain", "post", "wgap", "write", "commit", "next"]
NAMES = "t_start t_ldst t_loaded t_cload t_ckick t_cwait t_ccomp t_cdrain t_cdone t_post t_postv t_wstart t_wdone t_commit".split()


def parse(path):
    tiles, layers, cfg = [], {}, None
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = re.sub(r"^#\s*", "", line.strip())
            p = line.split(",")
            if p[0] == "CONFIG":
                cfg = (int(p[1]), int(p[2]))
            elif p[0] == "LAYER":
                layers[int(p[1])] = (int(p[2]), int(p[3]))
            elif p[0] == "TILE" and len(p) == 1 + 5 + len(NAMES) + 1:
                v = [int(x) for x in p[1:]]
                tiles.append({"layer": v[0], "idx": v[1], "K": v[2], "ru": v[3], "cu": v[4],
                              "t": dict(zip(NAMES, v[5:5 + len(NAMES)])), "n_wr": v[5 + len(NAMES)]})
    return cfg, layers, tiles


def phases(t, nxt):
    ts = t["t"]
    out = {"load": ts["t_loaded"] - ts["t_start"], "launch": ts["t_cload"] - ts["t_loaded"],
           "preload": ts["t_ccomp"] - ts["t_cload"], "compute": ts["t_cdrain"] - ts["t_ccomp"],
           "drain": ts["t_post"] - ts["t_cdrain"], "post": ts["t_postv"] - ts["t_post"],
           "wgap": ts["t_wstart"] - ts["t_postv"], "write": ts["t_wdone"] - ts["t_wstart"],
           "commit": ts["t_commit"] - ts["t_wdone"]}
    out["next"] = (nxt["t"]["t_start"] - ts["t_commit"]) if nxt else None
    return out


def lstsq(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return my, 0.0
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    return my - b * mx, b


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "rtl/cnn1d/sim/out/tb_tile_timing.log"
    cfg, layers, tiles = parse(path)
    if not tiles:
        sys.exit(f"khong co dong TILE trong {path}")
    print(f"mang {cfg[0]}x{cfg[1]}" if cfg else "mang ?", f"- {len(tiles)} o\n")
    for L in sorted({t["layer"] for t in tiles}):
        lt = [t for t in tiles if t["layer"] == L]
        rows = []
        for i, t in enumerate(lt):
            nxt = lt[i + 1] if i + 1 < len(lt) else None
            ph = phases(t, nxt)
            tau = sum(ph.values()) if nxt else None
            base = t["K"] * max(t["ru"], t["cu"]) + (t["K"] + t["ru"] + t["cu"])
            rows.append((t, ph, tau, base))
        tot_cyc = layers.get(L, (None, None))[0]
        print(f"=== layer {L}: {len(lt)} o, {tot_cyc} chu ky" if tot_cyc else f"=== layer {L}: {len(lt)} o")
        # nhom theo (ru, cu)
        groups = defaultdict(list)
        for r in rows:
            groups[(r[0]["ru"], r[0]["cu"])].append(r)
        hdr = f"{'ru x cu':>8} {'n':>4} {'n_wr':>5} " + " ".join(f"{p:>7}" for p in PHASES) + f" {'tau':>7} {'base':>6} {'delta':>6} {'d-nwr':>6}"
        print(hdr)
        for key in sorted(groups, key=lambda k: -groups[k][0][0]["n_wr"]):
            g = [r for r in groups[key] if r[2] is not None]
            if not g:
                continue
            avg = {p: sum(r[1][p] for r in g) / len(g) for p in PHASES}
            tau = sum(r[2] for r in g) / len(g)
            base = g[0][3]
            nwr = g[0][0]["n_wr"]
            delta = tau - base
            print(f"{key[0]:>4}x{key[1]:<3} {len(groups[key]):>4} {nwr:>5} " + " ".join(f"{avg[p]:>7.1f}" for p in PHASES)
                  + f" {tau:>7.1f} {base:>6} {delta:>6.1f} {delta - nwr:>6.1f}")
        full = [r for r in rows if r[2] is not None]
        mean_delta = sum(r[2] - r[3] for r in full) / len(full)
        # nap: load - K*max(ru,cu)
        load_err = {r[1]["load"] - r[0]["K"] * max(r[0]["ru"], r[0]["cu"]) for r in full}
        a, b = lstsq([r[0]["n_wr"] for r in full], [r[2] - r[3] for r in full])
        print(f"delta trung binh = {mean_delta:.1f}; fit delta = {a:.1f} + {b:.2f} * n_wr; "
              f"load - K*max(ru,cu) thuoc {sorted(load_err)}")
        if tot_cyc:
            ssum = sum(r[2] for r in full) + (rows[-1][1]["load"] + sum(rows[-1][1][p] for p in PHASES[1:-1]))
            print(f"tong chu ky layer {tot_cyc}; tong tau cac o (o cuoi tinh den commit) = {ssum}; "
                  f"phan ngoai cac o = {tot_cyc - ssum}")
        print()


if __name__ == "__main__":
    main()
