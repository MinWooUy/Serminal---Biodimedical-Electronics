#!/usr/bin/env bash
# Chạy một testbench bằng ModelSim (Intel FPGA Edition).
#   sim/run_tb.sh tb_conv1d_tile_top
#   sim/run_tb.sh all            # chạy mọi testbench trong tb/
# Đổi đường dẫn ModelSim bằng biến MODELSIM nếu cài chỗ khác.
set -euo pipefail
cd "$(dirname "$0")/.."

MSIM=${MODELSIM:-/c/intelFPGA_lite/18.1/modelsim_ase/win32aloem}
OUT=sim/out
mkdir -p "$OUT"
[ -d "$OUT/work" ] || "$MSIM/vlib" "$OUT/work" > /dev/null

run_one() {
	local tb=$1
	if ! "$MSIM/vlog" -quiet -work "$OUT/work" -f filelist.f "tb/$tb.v" > "$OUT/$tb.vlog.log"; then
		echo "$tb: LỖI BIÊN DỊCH (log: $OUT/$tb.vlog.log)"; return 1
	fi
	if ! "$MSIM/vsim" -c -lib "$OUT/work" -l "$OUT/$tb.log" -wlf "$OUT/$tb.wlf" \
		-do "run -all; quit -f" "$tb" > /dev/null; then
		echo "$tb: LỖI MÔ PHỎNG (log: $OUT/$tb.log)"; return 1
	fi
	local fails
	fails=$(grep -c -E '\[FAIL\]|FAIL:|FAILED|TIMEOUT|ERROR' "$OUT/$tb.log" || true)
	echo "$tb: PASS=$(grep -c 'PASS' "$OUT/$tb.log" || true) FAIL=$fails (log: $OUT/$tb.log)"
}

if [ "${1:-}" = "all" ]; then
	for f in tb/*.v; do run_one "$(basename "$f" .v)" || true; done
else
	run_one "${1:?Cần tên testbench, ví dụ: tb_conv1d_tile_top}"
fi
