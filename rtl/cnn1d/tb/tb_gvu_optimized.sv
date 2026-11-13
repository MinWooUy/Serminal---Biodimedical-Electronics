`timescale 1ns / 1ps
import gvu_config_pkg::*;

module tb_gvu_optimized;
    localparam int LANES = 8;
    localparam int DATA_WIDTH = 16;
    localparam int ACC_WIDTH = 32;

    logic clk = 1'b0;
    logic rst_n = 1'b0;
    logic valid_in = 1'b0;
    logic ready_in;
    logic [7:0] opcode = OP_ADD;
    logic [LANES-1:0][DATA_WIDTH-1:0] operand_a = '0;
    logic [LANES-1:0][DATA_WIDTH-1:0] operand_b = '0;
    logic valid_out;
    logic [LANES-1:0][ACC_WIDTH-1:0] result;
    logic [LANES-1:0][ACC_WIDTH-1:0] quantized;

    always #5 clk = ~clk;

    gvu_functional_unit #(
        .LANES(LANES), .DATA_WIDTH(DATA_WIDTH), .ACC_WIDTH(ACC_WIDTH), .FU_COUNT(1)
    ) dut (
        .clk(clk), .rst_n(rst_n), .valid_in(valid_in), .ready_in(ready_in),
        .opcode(opcode), .operand_a(operand_a), .operand_b(operand_b),
        .valid_out(valid_out), .ready_out(1'b1), .result(result)
    );

    gvu_runtime_quantize #(
        .LANES(LANES), .DATA_WIDTH(DATA_WIDTH), .ACC_WIDTH(ACC_WIDTH),
        .SUPPORT_SATURATION(1'b1), .SUPPORT_ROUNDING(1'b1)
    ) quantizer (
        .data_in(result), .shift(6'd1), .rounding_mode(2'd1),
        .clamp_min(-16'sd128), .clamp_max(16'sd127), .data_out(quantized)
    );

    initial begin : test
        integer lane;
        repeat (2) @(negedge clk);
        rst_n = 1'b1;

        for (lane = 0; lane < LANES; lane = lane + 1) begin
            operand_a[lane] = lane + 1;
            operand_b[lane] = 2;
        end

        opcode = OP_ADD;
        valid_in = 1'b1;
        @(posedge clk);
        #1;
        if (!valid_out) $fatal(1, "optimized GVU add did not produce output");
        for (lane = 0; lane < LANES; lane = lane + 1)
            if (result[lane] !== lane + 3) $fatal(1, "add failed at lane %0d", lane);

        opcode = OP_MUL;
        for (lane = 0; lane < LANES; lane = lane + 1)
            operand_a[lane] = lane + 3;
        @(posedge clk);
        #1;
        for (lane = 0; lane < LANES; lane = lane + 1)
            if (result[lane] !== (lane + 3) * 2) $fatal(1, "multiply failed at lane %0d", lane);

        #1;
        if (quantized[0] !== 32'd3 || quantized[7] !== 32'd10)
            $fatal(1, "round-shift quantization failed");

        operand_a[0] = 16'sd32767;
        operand_b[0] = 16'sd32767;
        opcode = OP_MUL;
        @(posedge clk);
        #1;
        if (quantized[0] !== 32'd127) $fatal(1, "saturation failed");

        $display("PASS optimized GVU profile: %0d lanes, %0d-bit data, %0d registers, quantization enabled", LANES, DATA_WIDTH, NUM_VECTOR_REGS);
        $finish;
    end
endmodule