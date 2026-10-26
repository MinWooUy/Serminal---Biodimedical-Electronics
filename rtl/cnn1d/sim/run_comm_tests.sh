#!/usr/bin/env bash
# Icarus Verilog: sim/run_comm_tests.sh
# Overrides: IVERILOG, VVP, IVL_BASE (for an extracted Icarus package).
set -euo pipefail
cd "$(dirname "$0")/.."
out=$(mktemp -d "${TMPDIR:-/tmp}/ecg-comm-tests.XXXXXX")
trap 'rm -rf "$out"' EXIT
compiler=${IVERILOG:-iverilog}
runtime=${VVP:-vvp}
flags=(-g2012)
if [[ -n ${IVL_BASE:-} ]]; then flags+=(-B "$IVL_BASE"); fi
for tb in tb_crc16_ccitt tb_uart tb_packet_rx tb_uart_crc_top; do
    "$compiler" "${flags[@]}" -s "$tb" -o "$out/$tb.vvp" -f filelist.f "tb/$tb.v"
    "$runtime" "$out/$tb.vvp"
done
PYTHONDONTWRITEBYTECODE=1 python3 comm/host/test_ecg_host.py
