`timescale 1ns / 1ps
import gvu_config_pkg::*;

module tb_gvu_runtime_config;
    localparam int LANES = 4;
    localparam int WIDTH = 16;
    localparam int ADDR_WIDTH = 8;
    localparam int LEN_WIDTH = $clog2(LANES + 1);
    logic clk = 1'b0;
    logic rst_n = 1'b0;
    logic cfg_valid = 1'b0;
    logic cfg_ready;
    logic [7:0] cfg_opcode = 8'h00;
    logic [2:0] cfg_data_type = DATA_TYPE_INT16;
    logic [LEN_WIDTH-1:0] cfg_vector_length = 3'd4;
    logic signed [ADDR_WIDTH-1:0] cfg_stride = 2'sd1;
    logic signed [ADDR_WIDTH-1:0] cfg_offset = '0;
    logic [5:0] cfg_shift = 6'd3;
    logic [1:0] cfg_rounding_mode = 2'd0;
    logic signed [WIDTH-1:0] cfg_clamp_min = -16'sd128;
    logic signed [WIDTH-1:0] cfg_clamp_max = 16'sd127;
    logic cfg_accumulate = 1'b1;
    logic [7:0] opcode;
    logic [2:0] data_type;
    logic [LEN_WIDTH-1:0] vector_length;
    logic signed [ADDR_WIDTH-1:0] stride, offset;
    logic [5:0] shift;
    logic [1:0] rounding_mode;
    logic signed [WIDTH-1:0] clamp_min, clamp_max;
    logic accumulate, cfg_error, unsupported_type;
    always #5 clk = ~clk;

    gvu_runtime_config #(
        .DATA_WIDTH(WIDTH), .MAX_VECTOR_WIDTH(64), .LANES(LANES), .ADDRESS_WIDTH(ADDR_WIDTH),
        .SUPPORT_INT8(1'b1), .SUPPORT_INT16(1'b1), .SUPPORT_INT32(1'b0),
        .SUPPORT_FP16(1'b0), .SUPPORT_FP32(1'b0)
    ) dut (
        .clk(clk), .rst_n(rst_n), .cfg_valid(cfg_valid), .cfg_ready(cfg_ready),
        .cfg_opcode(cfg_opcode), .cfg_data_type(cfg_data_type), .cfg_vector_length(cfg_vector_length),
        .cfg_stride(cfg_stride), .cfg_offset(cfg_offset), .cfg_shift(cfg_shift),
        .cfg_rounding_mode(cfg_rounding_mode), .cfg_clamp_min(cfg_clamp_min), .cfg_clamp_max(cfg_clamp_max),
        .cfg_accumulate(cfg_accumulate), .opcode(opcode), .data_type(data_type),
        .vector_length(vector_length), .stride(stride), .offset(offset), .shift(shift),
        .rounding_mode(rounding_mode), .clamp_min(clamp_min), .clamp_max(clamp_max),
        .accumulate(accumulate), .cfg_error(cfg_error), .unsupported_type(unsupported_type)
    );

    initial begin
        #1;
        if (cfg_error || unsupported_type) $fatal(1, "valid INT16 runtime config rejected");
        repeat (2) @(negedge clk);
        rst_n = 1'b1;
        cfg_valid = 1'b1;
        @(posedge clk);
        #1;
        cfg_valid = 1'b0;
        if (opcode !== cfg_opcode || data_type !== DATA_TYPE_INT16 || vector_length !== 3'd4 ||
            stride !== 2'sd1 || shift !== 6'd3 || !accumulate) $fatal(1, "runtime config was not committed");

        cfg_data_type = DATA_TYPE_FP32;
        #1;
        if (!cfg_error || !unsupported_type) $fatal(1, "unsupported FP32 was accepted");
        cfg_data_type = DATA_TYPE_INT16;
        cfg_vector_length = 3'd0;
        #1;
        if (!cfg_error) $fatal(1, "zero vector length was accepted");
        $display("PASS runtime capability and control checks");
        $finish;
    end
endmodule
