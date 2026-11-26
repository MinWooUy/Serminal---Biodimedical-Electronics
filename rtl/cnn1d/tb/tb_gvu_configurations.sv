`timescale 1ns / 1ps
import gvu_config_pkg::*;

module tb_gvu_configurations;
    logic clk = 1'b0;
    logic rst_n = 1'b0;
    always #5 clk = ~clk;

    logic v_small, v_mid, v_large;
    logic ready_small, ready_mid, ready_large;
    logic out_small, out_mid, out_large;
    logic [1:0][15:0] small_result;
    logic [3:0][31:0] mid_result;
    logic [7:0][63:0] large_result;
    logic [1:0][7:0] small_a, small_b;
    logic [3:0][15:0] mid_a, mid_b;
    logic [7:0][15:0] large_a, large_b;

    gvu_functional_unit #(.LANES(2), .DATA_WIDTH(8), .ACC_WIDTH(16), .FU_COUNT(1)) small_fu (
        .clk(clk), .rst_n(rst_n), .valid_in(v_small), .ready_in(ready_small), .opcode(8'h00),
        .operand_a(small_a), .operand_b(small_b), .valid_out(out_small), .ready_out(1'b1), .result(small_result));
    gvu_functional_unit #(.LANES(4), .DATA_WIDTH(16), .ACC_WIDTH(32), .FU_COUNT(2)) mid_fu (
        .clk(clk), .rst_n(rst_n), .valid_in(v_mid), .ready_in(ready_mid), .opcode(8'h02),
        .operand_a(mid_a), .operand_b(mid_b), .valid_out(out_mid), .ready_out(1'b1), .result(mid_result));
    gvu_functional_unit #(.LANES(8), .DATA_WIDTH(16), .ACC_WIDTH(64), .FU_COUNT(4)) large_fu (
        .clk(clk), .rst_n(rst_n), .valid_in(v_large), .ready_in(ready_large), .opcode(8'h01),
        .operand_a(large_a), .operand_b(large_b), .valid_out(out_large), .ready_out(1'b1), .result(large_result));

    initial begin : test
        integer index;
        v_small = 1'b0;
        v_mid = 1'b0;
        v_large = 1'b0;
        small_a = '0;
        small_b = '0;
        mid_a = '0;
        mid_b = '0;
        large_a = '0;
        large_b = '0;
        repeat (2) @(negedge clk);
        rst_n = 1'b1;

        small_a[0] = 8'h7f;
        small_a[1] = -8'sd3;
        small_b[0] = 8'd1;
        small_b[1] = 8'd5;
        for (index = 0; index < 4; index = index + 1) begin
            mid_a[index] = 16'd3;
            mid_b[index] = 16'd4;
        end
        for (index = 0; index < 8; index = index + 1) begin
            large_a[index] = 16'd9;
            large_b[index] = 16'd2;
        end
        v_small = 1'b1;
        v_mid = 1'b1;
        v_large = 1'b1;
        @(posedge clk);
        #1;
        v_small = 1'b0;
        v_mid = 1'b0;
        v_large = 1'b0;
        if (!out_small || small_result[0] !== 16'd128 || $signed(small_result[1]) !== 16'sd2) $fatal(1, "small configuration failed: valid=%b r0=%0d r1=%0d", out_small, small_result[0], $signed(small_result[1]));
        if (!out_mid || mid_result[0] !== 32'd12 || mid_result[3] !== 32'd12) $fatal(1, "medium configuration failed: valid=%b r0=%0d", out_mid, mid_result[0]);
        if (!out_large || $signed(large_result[0]) !== 64'sd7 || $signed(large_result[7]) !== 64'sd7) $fatal(1, "extended configuration failed: valid=%b r0=%0d", out_large, $signed(large_result[0]));

        $display("PASS three elaboration configurations");
        $finish;
    end
endmodule
