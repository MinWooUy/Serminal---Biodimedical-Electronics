#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
rtl="$root/rtl/cnn1d/GVU"
tb="$root/rtl/cnn1d/tb"
compiler=${IVERILOG:-iverilog}

run_test() {
    local name=$1
    shift
    local output
    output=$(mktemp "/tmp/${name}.XXXXXX")
    trap 'rm -f "$output"' RETURN
    "$compiler" -g2012 -Wall -s "$name" -o "$output" "$@"
    vvp "$output"
}

run_test tb_gvu_optimized "$rtl/gvu_config_pkg.sv" "$rtl/gvu_architecture.sv" "$rtl/gvu_top.sv" "$tb/tb_gvu_optimized.sv"
run_test tb_gvu_architecture "$rtl/gvu_config_pkg.sv" "$rtl/gvu_architecture.sv" "$tb/tb_gvu_architecture.sv"
run_test tb_gvu_modules "$rtl/gvu_config_pkg.sv" "$rtl/gvu_architecture.sv" "$tb/tb_gvu_modules.sv"
run_test tb_gvu_configurations "$rtl/gvu_config_pkg.sv" "$rtl/gvu_architecture.sv" "$tb/tb_gvu_configurations.sv"
run_test tb_gvu_runtime_config "$rtl/gvu_config_pkg.sv" "$rtl/gvu_architecture.sv" "$rtl/gvu_top.sv" "$tb/tb_gvu_runtime_config.sv"