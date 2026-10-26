`timescale 1ns / 1ps
import gvu_config_pkg::*;

module gvu_runtime_quantize #(
    parameter int LANES = gvu_config_pkg::VECTOR_LANES,
    parameter int DATA_WIDTH = gvu_config_pkg::DATA_WIDTH,
    parameter int ACC_WIDTH = gvu_config_pkg::ACC_WIDTH,
    parameter bit SUPPORT_SATURATION = gvu_config_pkg::SUPPORT_SATURATION,
    parameter bit SUPPORT_ROUNDING = gvu_config_pkg::SUPPORT_ROUNDING
) (
    input logic [LANES-1:0][ACC_WIDTH-1:0] data_in,
    input logic [5:0] shift,
    input logic [1:0] rounding_mode,
    input logic signed [DATA_WIDTH-1:0] clamp_min,
    input logic signed [DATA_WIDTH-1:0] clamp_max,
    output logic [LANES-1:0][ACC_WIDTH-1:0] data_out
);
    function automatic logic [ACC_WIDTH-1:0] quantize_one(
        input logic [ACC_WIDTH-1:0] value,
        input logic [5:0] shift_value,
        input logic [1:0] round_mode,
        input logic signed [DATA_WIDTH-1:0] minimum,
        input logic signed [DATA_WIDTH-1:0] maximum
    );
        logic signed [ACC_WIDTH-1:0] work;
        logic signed [ACC_WIDTH-1:0] min_extended;
        logic signed [ACC_WIDTH-1:0] max_extended;
        begin
            work = $signed(value);
            min_extended = $signed(minimum);
            max_extended = $signed(maximum);
            if (SUPPORT_ROUNDING && (round_mode != 0) && (shift_value != 0)) begin
                work = work + ({{(ACC_WIDTH-1){1'b0}}, 1'b1} <<< (shift_value - 1));
            end
            if (shift_value != 0) work = work >>> shift_value;
            if (SUPPORT_SATURATION) begin
                if (work > max_extended) work = max_extended;
                else if (work < min_extended) work = min_extended;
            end
            quantize_one = work;
        end
    endfunction

    integer lane;
    always_comb begin
        for (lane = 0; lane < LANES; lane = lane + 1) begin
            data_out[lane] = quantize_one(data_in[lane], shift, rounding_mode, clamp_min, clamp_max);
        end
    end
endmodule

module gvu_runtime_config #(
    parameter int DATA_WIDTH = gvu_config_pkg::DATA_WIDTH,
    parameter int MAX_VECTOR_WIDTH = gvu_config_pkg::MAX_VECTOR_WIDTH,
    parameter int LANES = gvu_config_pkg::VECTOR_LANES,
    parameter int OPCODE_WIDTH = gvu_config_pkg::OPCODE_WIDTH,
    parameter int ADDRESS_WIDTH = gvu_config_pkg::MEM_ADDR_WIDTH,
    parameter bit SUPPORT_INT8 = gvu_config_pkg::SUPPORT_INT8,
    parameter bit SUPPORT_INT16 = gvu_config_pkg::SUPPORT_INT16,
    parameter bit SUPPORT_INT32 = gvu_config_pkg::SUPPORT_INT32,
    parameter bit SUPPORT_FP16 = gvu_config_pkg::SUPPORT_FP16,
    parameter bit SUPPORT_FP32 = gvu_config_pkg::SUPPORT_FP32,
    parameter bit SUPPORT_SATURATION = gvu_config_pkg::SUPPORT_SATURATION,
    parameter bit SUPPORT_ROUNDING = gvu_config_pkg::SUPPORT_ROUNDING
) (
    input logic clk,
    input logic rst_n,
    input logic cfg_valid,
    output logic cfg_ready,
    input logic [OPCODE_WIDTH-1:0] cfg_opcode,
    input logic [2:0] cfg_data_type,
    input logic [($clog2(LANES + 1))-1:0] cfg_vector_length,
    input logic signed [ADDRESS_WIDTH-1:0] cfg_stride,
    input logic signed [ADDRESS_WIDTH-1:0] cfg_offset,
    input logic [5:0] cfg_shift,
    input logic [1:0] cfg_rounding_mode,
    input logic signed [DATA_WIDTH-1:0] cfg_clamp_min,
    input logic signed [DATA_WIDTH-1:0] cfg_clamp_max,
    input logic cfg_accumulate,
    output logic [OPCODE_WIDTH-1:0] opcode,
    output logic [2:0] data_type,
    output logic [($clog2(LANES + 1))-1:0] vector_length,
    output logic signed [ADDRESS_WIDTH-1:0] stride,
    output logic signed [ADDRESS_WIDTH-1:0] offset,
    output logic [5:0] shift,
    output logic [1:0] rounding_mode,
    output logic signed [DATA_WIDTH-1:0] clamp_min,
    output logic signed [DATA_WIDTH-1:0] clamp_max,
    output logic accumulate,
    output logic cfg_error,
    output logic unsupported_type
);
    assign cfg_ready = 1'b1;

    always_comb begin
        unsupported_type = 1'b0;
        case (cfg_data_type)
            DATA_TYPE_INT8: unsupported_type = !SUPPORT_INT8 || (DATA_WIDTH < 8);
            DATA_TYPE_INT16: unsupported_type = !SUPPORT_INT16 || (DATA_WIDTH < 16);
            DATA_TYPE_INT32: unsupported_type = !SUPPORT_INT32 || (DATA_WIDTH < 32);
            DATA_TYPE_FP16: unsupported_type = !SUPPORT_FP16;
            DATA_TYPE_FP32: unsupported_type = !SUPPORT_FP32;
            default: unsupported_type = 1'b1;
        endcase
        cfg_error = (cfg_vector_length == '0) || (cfg_vector_length > LANES) ||
                     (cfg_rounding_mode != 0 && !SUPPORT_ROUNDING) ||
                     ((cfg_clamp_min > cfg_clamp_max) && SUPPORT_SATURATION) ||
                     unsupported_type;
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            opcode <= OP_ADD;
            data_type <= DATA_TYPE_INT8;
            vector_length <= LANES;
            stride <= '0;
            offset <= '0;
            shift <= '0;
            rounding_mode <= '0;
            clamp_min <= '0;
            clamp_max <= {DATA_WIDTH{1'b1}};
            accumulate <= 1'b0;
        end else if (cfg_valid && cfg_ready && !cfg_error) begin
            opcode <= cfg_opcode;
            data_type <= cfg_data_type;
            vector_length <= cfg_vector_length;
            stride <= cfg_stride;
            offset <= cfg_offset;
            shift <= cfg_shift;
            rounding_mode <= cfg_rounding_mode;
            clamp_min <= cfg_clamp_min;
            clamp_max <= cfg_clamp_max;
            accumulate <= cfg_accumulate;
        end
    end
endmodule

module gvu_top #(
    parameter int DATA_WIDTH = gvu_config_pkg::DATA_WIDTH,
    parameter int MAX_VECTOR_WIDTH = gvu_config_pkg::MAX_VECTOR_WIDTH,
    parameter int LANES = gvu_config_pkg::VECTOR_LANES,
    parameter int REGISTER_COUNT = gvu_config_pkg::NUM_VECTOR_REGS,
    parameter int NUM_SCRATCHPAD_BANKS = gvu_config_pkg::NUM_SCRATCHPAD_BANKS,
    parameter int SCRATCHPAD_DEPTH = gvu_config_pkg::SCRATCHPAD_DEPTH,
    parameter int ACC_WIDTH = gvu_config_pkg::ACC_WIDTH,
    parameter int PSUM_WIDTH = gvu_config_pkg::PSUM_WIDTH,
    parameter int MAX_OUTSTANDING = gvu_config_pkg::MAX_OUTSTANDING,
    parameter int PIPELINE_STAGES = gvu_config_pkg::PIPELINE_STAGES,
    parameter int MAX_INFLIGHT_OPS = gvu_config_pkg::MAX_INFLIGHT_OPS,
    parameter int OPCODE_WIDTH = gvu_config_pkg::OPCODE_WIDTH,
    parameter int ADDRESS_WIDTH = gvu_config_pkg::MEM_ADDR_WIDTH,
    parameter bit SUPPORT_INT8 = gvu_config_pkg::SUPPORT_INT8,
    parameter bit SUPPORT_INT16 = gvu_config_pkg::SUPPORT_INT16,
    parameter bit SUPPORT_INT32 = gvu_config_pkg::SUPPORT_INT32,
    parameter bit SUPPORT_FP16 = gvu_config_pkg::SUPPORT_FP16,
    parameter bit SUPPORT_FP32 = gvu_config_pkg::SUPPORT_FP32,
    parameter bit SUPPORT_SATURATION = gvu_config_pkg::SUPPORT_SATURATION,
    parameter bit SUPPORT_ROUNDING = gvu_config_pkg::SUPPORT_ROUNDING,
    parameter int FU_COUNT = 1,
    parameter int REG_ADDR_WIDTH = $clog2((REGISTER_COUNT < 2) ? 2 : REGISTER_COUNT)
) (
    input logic clk,
    input logic rst_n,
    input logic instruction_valid,
    input logic [OPCODE_WIDTH-1:0] instruction_opcode,
    input logic [REG_ADDR_WIDTH-1:0] instruction_rd,
    input logic [REG_ADDR_WIDTH-1:0] instruction_rs1,
    input logic [REG_ADDR_WIDTH-1:0] instruction_rs2,
    input logic [LANES-1:0] instruction_lane_mask,
    input logic [LANES-1:0][DATA_WIDTH-1:0] load_data,
    output logic load_ready,
    input logic cfg_valid,
    output logic cfg_ready,
    input logic [OPCODE_WIDTH-1:0] cfg_opcode,
    input logic [2:0] cfg_data_type,
    input logic [$clog2(LANES + 1)-1:0] cfg_vector_length,
    input logic signed [ADDRESS_WIDTH-1:0] cfg_stride,
    input logic signed [ADDRESS_WIDTH-1:0] cfg_offset,
    input logic [5:0] cfg_shift,
    input logic [1:0] cfg_rounding_mode,
    input logic signed [DATA_WIDTH-1:0] cfg_clamp_min,
    input logic signed [DATA_WIDTH-1:0] cfg_clamp_max,
    input logic cfg_accumulate,
    output logic cfg_error,
    output logic unsupported_type,
    output logic [LANES-1:0][ACC_WIDTH-1:0] result,
    output logic result_valid,
    output logic sram_c_we,
    output logic [ADDRESS_WIDTH-1:0] sram_c_address,
    output logic [LANES-1:0][ACC_WIDTH-1:0] sram_c_data
);
    logic [OPCODE_WIDTH-1:0] runtime_opcode;
    logic [2:0] runtime_data_type;
    logic [$clog2(LANES + 1)-1:0] runtime_vector_length;
    logic signed [ADDRESS_WIDTH-1:0] runtime_stride, runtime_offset;
    logic [5:0] runtime_shift;
    logic [1:0] runtime_rounding_mode;
    logic signed [DATA_WIDTH-1:0] runtime_clamp_min, runtime_clamp_max;
    logic runtime_accumulate;
    logic [LANES-1:0][ACC_WIDTH-1:0] raw_result;
    logic [LANES-1:0][ACC_WIDTH-1:0] raw_sram_data;
    logic [LANES-1:0][ACC_WIDTH-1:0] quantized_result;

    gvu_runtime_quantize #(
        .LANES(LANES), .DATA_WIDTH(DATA_WIDTH), .ACC_WIDTH(ACC_WIDTH),
        .SUPPORT_SATURATION(SUPPORT_SATURATION), .SUPPORT_ROUNDING(SUPPORT_ROUNDING)
    ) quantize (
        .data_in(raw_result), .shift(runtime_shift), .rounding_mode(runtime_rounding_mode),
        .clamp_min(runtime_clamp_min), .clamp_max(runtime_clamp_max), .data_out(quantized_result)
    );

    gvu_runtime_config #(
        .DATA_WIDTH(DATA_WIDTH), .MAX_VECTOR_WIDTH(MAX_VECTOR_WIDTH), .LANES(LANES),
        .OPCODE_WIDTH(OPCODE_WIDTH), .ADDRESS_WIDTH(ADDRESS_WIDTH),
        .SUPPORT_INT8(SUPPORT_INT8), .SUPPORT_INT16(SUPPORT_INT16), .SUPPORT_INT32(SUPPORT_INT32),
        .SUPPORT_FP16(SUPPORT_FP16), .SUPPORT_FP32(SUPPORT_FP32),
        .SUPPORT_SATURATION(SUPPORT_SATURATION), .SUPPORT_ROUNDING(SUPPORT_ROUNDING)
    ) runtime_cfg (
        .clk(clk), .rst_n(rst_n), .cfg_valid(cfg_valid), .cfg_ready(cfg_ready),
        .cfg_opcode(cfg_opcode), .cfg_data_type(cfg_data_type), .cfg_vector_length(cfg_vector_length),
        .cfg_stride(cfg_stride), .cfg_offset(cfg_offset), .cfg_shift(cfg_shift),
        .cfg_rounding_mode(cfg_rounding_mode), .cfg_clamp_min(cfg_clamp_min), .cfg_clamp_max(cfg_clamp_max),
        .cfg_accumulate(cfg_accumulate), .opcode(runtime_opcode), .data_type(runtime_data_type),
        .vector_length(runtime_vector_length), .stride(runtime_stride), .offset(runtime_offset),
        .shift(runtime_shift), .rounding_mode(runtime_rounding_mode), .clamp_min(runtime_clamp_min),
        .clamp_max(runtime_clamp_max), .accumulate(runtime_accumulate), .cfg_error(cfg_error),
        .unsupported_type(unsupported_type)
    );

    gvu #(
        .DATA_WIDTH(DATA_WIDTH), .MAX_VECTOR_WIDTH(MAX_VECTOR_WIDTH), .LANES(LANES),
        .REGISTER_COUNT(REGISTER_COUNT), .NUM_SCRATCHPAD_BANKS(NUM_SCRATCHPAD_BANKS),
        .SCRATCHPAD_DEPTH(SCRATCHPAD_DEPTH), .ACC_WIDTH(ACC_WIDTH), .PSUM_WIDTH(PSUM_WIDTH),
        .MAX_OUTSTANDING(MAX_OUTSTANDING), .PIPELINE_STAGES(PIPELINE_STAGES),
        .MAX_INFLIGHT_OPS(MAX_INFLIGHT_OPS), .SUPPORT_INT8(SUPPORT_INT8),
        .SUPPORT_INT16(SUPPORT_INT16), .SUPPORT_INT32(SUPPORT_INT32), .SUPPORT_FP16(SUPPORT_FP16),
        .SUPPORT_FP32(SUPPORT_FP32), .SUPPORT_SATURATION(SUPPORT_SATURATION),
        .SUPPORT_ROUNDING(SUPPORT_ROUNDING), .FU_COUNT(FU_COUNT), .OPCODE_WIDTH(OPCODE_WIDTH),
        .ADDRESS_WIDTH(ADDRESS_WIDTH), .REGISTER_INDEX_WIDTH(REG_ADDR_WIDTH)
    ) datapath (
        .clk(clk), .rst_n(rst_n), .instruction_valid(instruction_valid && !cfg_error),
        .instruction_opcode(runtime_opcode), .instruction_rd(instruction_rd),
        .instruction_rs1(instruction_rs1), .instruction_rs2(instruction_rs2),
        .instruction_lane_mask(instruction_lane_mask), .load_data(load_data),
        .load_valid(1'b0), .load_ready(load_ready), .result(raw_result), .result_valid(result_valid),
        .sram_c_we(sram_c_we), .sram_c_address(sram_c_address), .sram_c_data(raw_sram_data)
    );
    assign result = quantized_result;
    assign sram_c_data = quantized_result;
endmodule
