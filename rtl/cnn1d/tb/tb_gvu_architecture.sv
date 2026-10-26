`timescale 1ns / 1ps

module tb_gvu_architecture;
    localparam int LANES = 2;
    localparam int DATA_WIDTH = 8;
    localparam int ACC_WIDTH = 16;

    logic clk = 1'b0;
    logic rst_n = 1'b0;
    logic valid_in = 1'b0;
    logic ready_in;
    logic [7:0] opcode = 8'h00;
    logic [LANES-1:0][DATA_WIDTH-1:0] operand_a = '0;
    logic [LANES-1:0][DATA_WIDTH-1:0] operand_b = '0;
    logic valid_out;
    logic [LANES-1:0][ACC_WIDTH-1:0] result;
    logic [34:0] instruction = '0;
    logic instruction_valid = 1'b1;
    logic [7:0] decoded_opcode;
    logic [4:0] decoded_rd;
    logic [4:0] decoded_rs1;
    logic [4:0] decoded_rs2;
    logic [11:0] decoded_imm;
    logic decoded_valid;
    logic decoded_illegal;

    always #5 clk = ~clk;

    gvu_functional_unit #(
        .LANES(LANES), .DATA_WIDTH(DATA_WIDTH), .ACC_WIDTH(ACC_WIDTH), .FU_COUNT(2)
    ) dut_fu (
        .clk(clk), .rst_n(rst_n), .valid_in(valid_in), .ready_in(ready_in), .opcode(opcode),
        .operand_a(operand_a), .operand_b(operand_b), .valid_out(valid_out), .ready_out(1'b1), .result(result)
    );

    gvu_instruction_decode_if dut_decode (
        .instruction(instruction), .valid_in(instruction_valid), .opcode(decoded_opcode), .rd(decoded_rd), .rs1(decoded_rs1),
        .rs2(decoded_rs2), .immediate(decoded_imm), .valid_out(decoded_valid), .illegal_instruction(decoded_illegal)
    );

    initial begin
        instruction = {12'habc, 5'd3, 5'd2, 5'd1, 8'h06};
        #1;
        if (decoded_opcode !== 8'h06 || decoded_rd !== 5'd1 || decoded_rs1 !== 5'd2 ||
            decoded_rs2 !== 5'd3 || decoded_imm !== 12'habc || !decoded_valid || decoded_illegal) $fatal(1, "decode failed");

        repeat (2) @(negedge clk);
        rst_n = 1'b1;
        operand_a[0] = 8'd7;
        operand_a[1] = 8'd9;
        operand_b[0] = 8'd5;
        operand_b[1] = 8'd3;
        opcode = 8'h00;
        @(negedge clk);
        valid_in = 1'b1;
        @(posedge clk);
        #1;
        if (!valid_out || result[0] !== 16'd12 || result[1] !== 16'd12) $fatal(1, "vector add failed");
        @(negedge clk);
        valid_in = 1'b0;

        operand_a[0] = 8'd6;
        operand_a[1] = 8'd4;
        operand_b[0] = 8'd2;
        operand_b[1] = 8'd3;
        opcode = 8'h02;
        @(negedge clk);
        valid_in = 1'b1;
        @(posedge clk);
        #1;
        if (!valid_out || result[0] !== 16'd12 || result[1] !== 16'd12) $fatal(1, "vector multiply failed");
        @(negedge clk);
        valid_in = 1'b0;

        $display("PASS tb_gvu_architecture");
        $finish;
    end
endmodule