module wallace_tree_multiplier#(
  parameter DATA_W = 8
)(
  input wire[DATA_W-1:0] i_a,
  input wire[DATA_W-1:0] i_b,

  output wire[(2*DATA_W)-1:0] o_p
);
  // step 1: generate pp matrix
  wire[(DATA_W*DATA_W)-1:0] pp;

  baugh_wooley_partial_product #(
    .DATA_W(DATA_W)
    ) bw_pp(
    .a(i_a), .b(i_b), .pp(pp)
  );

  // step 2: reduce rows using Wallace tree
  wire[(2*DATA_W)-1:0] sum, carry;

  wallace_tree_reducer #(
    .DATA_W(DATA_W)
    ) wtr(
    .pp(pp), .sum(sum), .carry(carry)
  );

  // step 3: final addition using carry lookahead adder
  wire [(2*DATA_W)-1:0] r_p;

  final_carry_lookahead_adder #(
    .DATA_W(DATA_W)
    ) cla(
    .A(sum), .B(carry), .Cin(1'b0),
    .S(r_p), .Cout()
  );

  // step 4: inverter MSB
  assign o_p[(2*DATA_W-2):0] = r_p[(2*DATA_W-2):0];
  assign o_p[(2*DATA_W-1)] = ~r_p[(2*DATA_W)-1];
endmodule