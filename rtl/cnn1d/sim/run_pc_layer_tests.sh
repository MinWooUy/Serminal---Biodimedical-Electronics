#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

out=$(mktemp -d "${TMPDIR:-/tmp}/ecg-pc-layer-tests.XXXXXX")
trap 'rm -rf "$out"' EXIT
compiler=${IVERILOG:-iverilog}
runtime=${VVP:-vvp}
flags=(-g2012)
if [[ -n ${IVL_BASE:-} ]]; then flags+=(-B "$IVL_BASE"); fi

for tb in tb_pc_to_fpga_layer_flow tb_cnn_pipeline_controller; do
    "$compiler" "${flags[@]}" -s "$tb" -o "$out/$tb.vvp" -f filelist.f "tb/$tb.v"
    "$runtime" "$out/$tb.vvp"
done
