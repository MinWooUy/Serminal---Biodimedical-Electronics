`default_nettype none
module baugh_wooley_partial_product#(
  parameter DATA_W = 4
)(
  input wire[DATA_W-1:0] a,
  input wire[DATA_W-1:0] b,
  // Partial product
  output wire[(DATA_W*DATA_W)-1:0] pp
);
  // generate pp matrix
  genvar i,j;
  generate
    for(i = 0; i < DATA_W; i = i + 1) begin: gen_a
      for(j = 0; j < DATA_W; j = j + 1) begin: gen_b
        if(((i == DATA_W-1) && (j != DATA_W-1)) || ((i != DATA_W-1) && (j == DATA_W-1))) begin
          assign pp[i*DATA_W + j] = ~(a[i] & b[j]);
        end
        else begin
          assign pp[i*DATA_W + j] = a[i] & b[j];
        end
      end
    end
  endgenerate
endmodule
`default_nettype wire
