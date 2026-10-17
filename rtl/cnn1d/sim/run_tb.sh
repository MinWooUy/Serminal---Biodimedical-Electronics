#!/usr/bin/env bash
# Chạy testbench bằng ModelSim (Intel FPGA Edition), không cần mở Quartus.
# Hướng dẫn đầy đủ: docs/sim-flow.md
#
#   sim/run_tb.sh tb_conv1d_tile_top        # chạy 1 testbench (chế độ dòng lệnh)
#   sim/run_tb.sh ring_buffer               # tên module cũng được: tự tìm tb/tb_ring_buffer.v
#   sim/run_tb.sh all                       # chạy mọi testbench trong tb/ (regression)
#   sim/run_tb.sh --wave tb_ring_buffer     # chạy và ghi mọi tín hiệu vào sim/out/<tb>.wlf
#   sim/run_tb.sh --gui  tb_ring_buffer     # mở ModelSim GUI, thêm mọi tín hiệu vào cửa sổ wave
#
# Mã thoát: 0 khi mọi testbench PASS, 1 khi có testbench lỗi hoặc FAIL.
# Đổi đường dẫn ModelSim bằng biến MODELSIM nếu cài chỗ khác.
set -euo pipefail
cd "$(dirname "$0")/.."

MSIM=${MODELSIM:-/c/intelFPGA_lite/18.1/modelsim_ase/win32aloem}
OUT=sim/out
MODE=batch

case "${1:-}" in
	--wave) MODE=wave; shift ;;
	--gui)  MODE=gui;  shift ;;
esac

mkdir -p "$OUT"
[ -d "$OUT/work" ] || "$MSIM/vlib" "$OUT/work" > /dev/null

# Nhận tên testbench, tên module, hoặc đường dẫn file (.v)
resolve_tb() {
	local name
	name=$(basename "$1" .v)
	if [ -f "tb/$name.v" ]; then echo "$name"
	elif [ -f "tb/tb_$name.v" ]; then echo "tb_$name"
	else echo "Không tìm thấy testbench cho '$1' (cần tb/$name.v hoặc tb/tb_$name.v)" >&2; return 1
	fi
}

compile() {
	local tb=$1
	if ! "$MSIM/vlog" -quiet -work "$OUT/work" -f filelist.f "tb/$tb.v" > "$OUT/$tb.vlog.log"; then
		echo "$tb: LỖI BIÊN DỊCH (log: $OUT/$tb.vlog.log)"
		grep -E '\*\* Error' "$OUT/$tb.vlog.log" | head -5
		return 1
	fi
}

run_one() {
	local tb=$1 do_cmd="run -all; quit -f"
	compile "$tb" || return 1
	[ "$MODE" = wave ] && do_cmd="log -r /*; run -all; quit -f"
	if ! "$MSIM/vsim" -c -lib "$OUT/work" -l "$OUT/$tb.log" -wlf "$OUT/$tb.wlf" \
		-do "$do_cmd" "$tb" > /dev/null; then
		echo "$tb: LỖI MÔ PHỎNG (log: $OUT/$tb.log)"
		grep -E '\*\* (Error|Fatal)' "$OUT/$tb.log" | head -5
		return 1
	fi
	local pass fails
	pass=$(grep -c 'PASS' "$OUT/$tb.log" || true)
	fails=$(grep -c -E '\[FAIL\]|FAIL:|FAILED|TIMEOUT|ERROR' "$OUT/$tb.log" || true)
	echo "$tb: PASS=$pass FAIL=$fails (log: $OUT/$tb.log)"
	[ "$MODE" = wave ] && echo "  waveform: $MSIM/vsim -view $OUT/$tb.wlf"
	[ "$fails" -eq 0 ]
}

run_gui() {
	local tb=$1
	compile "$tb" || return 1
	echo "$tb: mở ModelSim GUI (đóng cửa sổ để quay lại terminal)"
	"$MSIM/vsim" -gui -lib "$OUT/work" -wlf "$OUT/$tb.wlf" \
		-do "log -r /*; add wave -r /*; run -all; wave zoom full" "$tb"
}

target="${1:?Cần tên testbench, ví dụ: sim/run_tb.sh tb_conv1d_tile_top}"

if [ "$target" = all ]; then
	[ "$MODE" = gui ] && { echo "--gui chỉ chạy được với 1 testbench"; exit 1; }
	failed=0
	for f in tb/*.v; do run_one "$(basename "$f" .v)" || failed=$((failed + 1)); done
	echo "----"
	if [ "$failed" -eq 0 ]; then echo "REGRESSION: tất cả PASS"; else echo "REGRESSION: $failed testbench lỗi"; exit 1; fi
else
	tb=$(resolve_tb "$target")
	if [ "$MODE" = gui ]; then run_gui "$tb"; else run_one "$tb"; fi
fi
