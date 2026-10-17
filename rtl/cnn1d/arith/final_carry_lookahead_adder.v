module final_carry_lookahead_adder #(
  parameter DATA_W = 8
)(
  input wire [(2*DATA_W)-1:0] A,
  input wire [(2*DATA_W)-1:0] B,
  input wire Cin,

  output wire [(2*DATA_W)-1:0] S,
  output wire Cout
);

  localparam CLA_W = 2 * DATA_W;

  generate
    // INT4
    if(CLA_W == 8) begin: gen_cla_4bit
      CarryLookaheadAdder_8_bit cla_8(
        .A(A), .B(B), .Cin(Cin),
        .S(S), .Cout(Cout)
      );
    end
    // INT8
    else if(CLA_W == 16) begin: gen_cla_8bit
      CarryLookaheadAdder_16_bit cla_16(
        .A(A), .B(B), .Cin(Cin),
        .S(S), .Cout(Cout)
      );
    end
    // INT16
    else if(CLA_W == 32) begin: gen_cla_16bit
      CarryLookaheadAdder_32_bit cla_32(
        .A(A), .B(B), .Cin(Cin),
        .S(S), .Cout(Cout)
      );
    end
  endgenerate
endmodule