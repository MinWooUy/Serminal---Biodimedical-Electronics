module wallace_tree_reducer#(
  parameter DATA_W = 4
)(
  input wire[(DATA_W*DATA_W)-1:0] pp,
  output wire[(2*DATA_W)-1:0] sum,
  output wire[(2*DATA_W)-1:0] carry
);
  // ----------------
  // INT8
  // ----------------
  generate
    if(DATA_W == 8) begin: gen_wallace_8bit
      wallace_tree_8bit wlt_8(
        .pp(pp), .sum(sum), .carry(carry)
      );
    end
  endgenerate
endmodule