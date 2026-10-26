`default_nettype none
module wallace_tree_8bit(
  input wire[63:0] pp,
  output wire[15:0] sum,
  output wire[15:0] carry
);
  wire s1_c2_0;
  wire c1_c3_0;
  full_adder fa_s1_c2_0(
    .A(pp[16]), .B(pp[9]), .CarryIn(pp[2]),
    .Y(s1_c2_0), .CarryOut(c1_c3_0)
  );

  wire s1_c3_0;
  wire c1_c4_0;
  full_adder fa_s1_c3_0(
    .A(pp[24]), .B(pp[17]), .CarryIn(pp[10]),
    .Y(s1_c3_0), .CarryOut(c1_c4_0)
  );

  wire s1_c4_0;
  wire c1_c5_0;
  full_adder fa_s1_c4_0(
    .A(pp[32]), .B(pp[25]), .CarryIn(pp[18]),
    .Y(s1_c4_0), .CarryOut(c1_c5_0)
  );

  wire s1_c5_0;
  wire c1_c6_0;
  wire s1_c5_1;
  wire c1_c6_1;
  full_adder fa_s1_c5_0(
    .A(pp[40]), .B(pp[33]), .CarryIn(pp[26]),
    .Y(s1_c5_0), .CarryOut(c1_c6_0)
  );
  full_adder fa_s1_c5_1(
    .A(pp[19]), .B(pp[12]), .CarryIn(pp[5]),
    .Y(s1_c5_1), .CarryOut(c1_c6_1)
  );

  wire s1_c6_0;
  wire c1_c7_0;
  wire s1_c6_1;
  wire c1_c7_1;
  full_adder fa_s1_c6_0(
    .A(pp[48]), .B(pp[41]), .CarryIn(pp[34]),
    .Y(s1_c6_0), .CarryOut(c1_c7_0)
  );
  full_adder fa_s1_c6_1(
    .A(pp[27]), .B(pp[20]), .CarryIn(pp[13]),
    .Y(s1_c6_1), .CarryOut(c1_c7_1)
  );

  wire s1_c7_0;
  wire c1_c8_0;
  wire s1_c7_1;
  wire c1_c8_1;
  full_adder fa_s1_c7_0(
    .A(pp[56]), .B(pp[49]), .CarryIn(pp[42]),
    .Y(s1_c7_0), .CarryOut(c1_c8_0)
  );
  full_adder fa_s1_c7_1(
    .A(pp[35]), .B(pp[28]), .CarryIn(pp[21]),
    .Y(s1_c7_1), .CarryOut(c1_c8_1)
  );

  wire s1_c8_0;
  wire c1_c9_0;
  wire s1_c8_1;
  wire c1_c9_1;
  full_adder fa_s1_c8_0(
    .A(pp[57]), .B(pp[50]), .CarryIn(pp[43]),
    .Y(s1_c8_0), .CarryOut(c1_c9_0)
  );
  full_adder fa_s1_c8_1(
    .A(pp[36]), .B(pp[29]), .CarryIn(pp[22]),
    .Y(s1_c8_1), .CarryOut(c1_c9_1)
  );

  wire s1_c9_0;
  wire c1_c10_0;
  wire s1_c9_1;
  wire c1_c10_1;
  full_adder fa_s1_c9_0(
    .A(pp[58]), .B(pp[51]), .CarryIn(pp[44]),
    .Y(s1_c9_0), .CarryOut(c1_c10_0)
  );
  full_adder fa_s1_c9_1(
    .A(pp[37]), .B(pp[30]), .CarryIn(pp[23]),
    .Y(s1_c9_1), .CarryOut(c1_c10_1)
  );

  wire s1_c10_0;
  wire c1_c11_0;
  full_adder fa_s1_c10_0(
    .A(pp[59]), .B(pp[52]), .CarryIn(pp[45]),
    .Y(s1_c10_0), .CarryOut(c1_c11_0)
  );

  wire s1_c11_0;
  wire c1_c12_0;
  full_adder fa_s1_c11_0(
    .A(pp[60]), .B(pp[53]), .CarryIn(pp[46]),
    .Y(s1_c11_0), .CarryOut(c1_c12_0)
  );

  wire s1_c12_0;
  wire c1_c13_0;
  full_adder fa_s1_c12_0(
    .A(pp[61]), .B(pp[54]), .CarryIn(pp[47]),
    .Y(s1_c12_0), .CarryOut(c1_c13_0)
  );
// ============================================================
// Stage 1 output column inventory
//
// C0 : pp[0]
// C1 : pp[8], pp[1]
// C2 : s1_c2_0
// C3 : s1_c3_0, pp[3], c1_c3_0
// C4 : s1_c4_0, pp[11], pp[4], c1_c4_0
// C5 : s1_c5_0, s1_c5_1, c1_c5_0
// C6 : s1_c6_0, s1_c6_1, pp[6], c1_c6_0, c1_c6_1
// C7 : s1_c7_0, s1_c7_1, pp[14], pp[7], c1_c7_0, c1_c7_1
// C8 : s1_c8_0, s1_c8_1, pp[15], 1'b1, c1_c8_0, c1_c8_1
// C9 : s1_c9_0, s1_c9_1, c1_c9_0, c1_c9_1
// C10: s1_c10_0, pp[38], pp[31], c1_c10_0, c1_c10_1
// C11: s1_c11_0, pp[39], c1_c11_0
// C12: s1_c12_0, c1_c12_0
// C13: pp[62], pp[55], c1_c13_0
// C14: pp[63]
// ============================================================

  wire s2_c3_0;
  wire c2_c4_0;
  full_adder fa_s2_c3_0(
    .A(s1_c3_0), .B(pp[3]), .CarryIn(c1_c3_0),
    .Y(s2_c3_0), .CarryOut(c2_c4_0)
  );

  wire s2_c4_0;
  wire c2_c5_0;
  full_adder fa_s2_c4_0(
    .A(pp[4]), .B(pp[11]), .CarryIn(s1_c4_0),
    .Y(s2_c4_0), .CarryOut(c2_c5_0)
  );

  wire s2_c5_0;
  wire c2_c6_0;
  full_adder fa_s2_c5_0(
    .A(s1_c5_0), .B(s1_c5_1), .CarryIn(c1_c5_0),
    .Y(s2_c5_0), .CarryOut(c2_c6_0)
  );

  wire s2_c6_0;
  wire c2_c7_0;
  full_adder fa_s2_c6_0(
    .A(s1_c6_0), .B(s1_c6_1), .CarryIn(pp[6]),
    .Y(s2_c6_0), .CarryOut(c2_c7_0)
  );

  wire s2_c7_0;
  wire c2_c8_0;
  wire s2_c7_1;
  wire c2_c8_1;
  full_adder fa_s2_c7_0(
    .A(s1_c7_0), .B(s1_c7_1), .CarryIn(pp[14]),
    .Y(s2_c7_0), .CarryOut(c2_c8_0)
  );
  full_adder fa_s2_c7_1(
    .A(pp[7]), .B(c1_c7_0), .CarryIn(c1_c7_1),
    .Y(s2_c7_1), .CarryOut(c2_c8_1)
  );

  wire s2_c8_0;
  wire c2_c9_0;
  wire s2_c8_1;
  wire c2_c9_1;
  full_adder fa_s2_c8_0(
    .A(s1_c8_0), .B(s1_c8_1), .CarryIn(pp[15]),
    .Y(s2_c8_0), .CarryOut(c2_c9_0)
  );
  full_adder fa_s2_c8_1(
    .A(1'b1), .B(c1_c8_0), .CarryIn(c1_c8_1), // Add correction
    .Y(s2_c8_1), .CarryOut(c2_c9_1)
  );

  wire s2_c9_0;
  wire c2_c10_0;
  full_adder fa_s2_c9_0(
    .A(s1_c9_0), .B(s1_c9_1), .CarryIn(c1_c9_0),
    .Y(s2_c9_0), .CarryOut(c2_c10_0)
  );

  wire s2_c10_0;
  wire c2_c11_0;
  full_adder fa_s2_c10_0(
    .A(s1_c10_0), .B(pp[38]), .CarryIn(pp[31]),
    .Y(s2_c10_0), .CarryOut(c2_c11_0)
  );

  wire s2_c11_0;
  wire c2_c12_0;
  full_adder fa_s2_c11_0(
    .A(s1_c11_0), .B(pp[39]), .CarryIn(c1_c11_0),
    .Y(s2_c11_0), .CarryOut(c2_c12_0)
  );

  wire s2_c13_0;
  wire c2_c14_0;
  full_adder fa_s2_c13_0(
    .A(pp[62]), .B(pp[55]), .CarryIn(c1_c13_0),
    .Y(s2_c13_0), .CarryOut(c2_c14_0)
  );

// ============================================================
// Stage 2 output column inventory
// C0 : pp[0]
// C1 : pp[8], pp[1]
// C2 : s1_c2_0
// C3 : s2_c3_0
// C4 : s2_c4_0, c1_c4_0, c2_c4_0
// C5 : s2_c5_0, c2_c5_0
// C6 : s2_c6_0, c1_c6_0, c1_c6_1, c2_c6_0
// C7 : s2_c7_0, s2_c7_1, c2_c7_0
// C8 : s2_c8_0, s2_c8_1, c2_c8_0, c2_c8_1
// C9 : s2_c9_0, c1_c9_1, c2_c9_0, c2_c9_1
// C10: s2_c10_0, c1_c10_0, c1_c10_1, c2_c10_0
// C11: s2_c11_0, c2_c11_0
// C12: s1_c12_0, c1_c12_0, c2_c12_0
// C13: s2_c13_0
// C14: pp[63], c2_c14_0
// ============================================================
  wire s3_c4_0;
  wire c3_c5_0;
  full_adder fa_s3_c4_0(
    .A(s2_c4_0), .B(c1_c4_0), .CarryIn(c2_c4_0),
    .Y(s3_c4_0), .CarryOut(c3_c5_0)
  );

  wire s3_c6_0;
  wire c3_c7_0;
  full_adder fa_s3_c6_0(
    .A(s2_c6_0), .B(c1_c6_0), .CarryIn(c1_c6_1),
    .Y(s3_c6_0), .CarryOut(c3_c7_0)
  );

  wire s3_c7_0;
  wire c3_c8_0;
  full_adder fa_s3_c7_0(
    .A(s2_c7_0), .B(s2_c7_1), .CarryIn(c2_c7_0),
    .Y(s3_c7_0), .CarryOut(c3_c8_0)
  );

  wire s3_c8_0;
  wire c3_c9_0;
  full_adder fa_s3_c8_0(
    .A(s2_c8_0), .B(s2_c8_1), .CarryIn(c2_c8_0),
    .Y(s3_c8_0), .CarryOut(c3_c9_0)
  );

  wire s3_c9_0;
  wire c3_c10_0;
  full_adder fa_s3_c9_0(
    .A(s2_c9_0), .B(c1_c9_1), .CarryIn(c2_c9_0),
    .Y(s3_c9_0), .CarryOut(c3_c10_0)
  );

  wire s3_c10_0;
  wire c3_c11_0;
  full_adder fa_s3_c10_0(
    .A(s2_c10_0), .B(c1_c10_0), .CarryIn(c1_c10_1),
    .Y(s3_c10_0), .CarryOut(c3_c11_0)
  );

  wire s3_c12_0;
  wire c3_c13_0;
  full_adder fa_s3_c12_0(
    .A(s1_c12_0), .B(c1_c12_0), .CarryIn(c2_c12_0),
    .Y(s3_c12_0), .CarryOut(c3_c13_0)
  );
// ============================================================
// Stage 3 output column inventory
// C0 : pp[0]
// C1 : pp[8], pp[1]
// C2 : s1_c2_0
// C3 : s2_c3_0
// C4 : s3_c4_0
// C5 : s2_c5_0, c2_c5_0, c3_c5_0
// C6 : s3_c6_0, c2_c6_0
// C7 : s3_c7_0, c3_c7_0
// C8 : s3_c8_0, c2_c8_1, c3_c8_0
// C9 : s3_c9_0, c2_c9_1, c3_c9_0
// C10: s3_c10_0, c2_c10_0, c3_c10_0
// C11: s2_c11_0, c2_c11_0, c3_c11_0
// C12: s3_c12_0
// C13: s2_c13_0, c3_c13_0
// C14: pp[63], c2_c14_0
// ============================================================
  wire s4_c5_0;
  wire c4_c6_0;
  full_adder fa_s4_c5_0(
    .A(s2_c5_0), .B(c2_c5_0), .CarryIn(c3_c5_0),
    .Y(s4_c5_0), .CarryOut(c4_c6_0)
  );

  wire s4_c6_0;
  wire c4_c7_0;
  half_adder ha_s4_c6_0(
    .A(s3_c6_0), .B(c2_c6_0),
    .S(s4_c6_0), .Cout(c4_c7_0)
  );

  wire s4_c8_0;
  wire c4_c9_0;
  full_adder fa_s4_c8_0(
    .A(s3_c8_0), .B(c2_c8_1), .CarryIn(c3_c8_0),
    .Y(s4_c8_0), .CarryOut(c4_c9_0)
  );

  wire s4_c9_0;
  wire c4_c10_0;
  full_adder fa_s4_c9_0(
    .A(s3_c9_0), .B(c2_c9_1), .CarryIn(c3_c9_0),
    .Y(s4_c9_0), .CarryOut(c4_c10_0)
  );

  wire s4_c10_0;
  wire c4_c11_0;
  full_adder fa_s4_c10_0(
    .A(s3_c10_0), .B(c2_c10_0), .CarryIn(c3_c10_0),
    .Y(s4_c10_0), .CarryOut(c4_c11_0)
  );

  wire s4_c11_0;
  wire c4_c12_0;
  full_adder fa_s4_c11_0(
    .A(s2_c11_0), .B(c2_c11_0), .CarryIn(c3_c11_0),
    .Y(s4_c11_0), .CarryOut(c4_c12_0)
  );
// ============================================================
// Stage 4 output column inventory
// C0 : pp[0]
// C1 : pp[8], pp[1]
// C2 : s1_c2_0
// C3 : s2_c3_0
// C4 : s3_c4_0
// C5 : s4_c5_0
// C6 : s4_c6_0, c4_c6_0
// C7 : s3_c7_0, c3_c7_0, c4_c7_0
// C8 : s4_c8_0
// C9 : s4_c9_0, c4_c9_0
// C10: s4_c10_0, c4_c10_0
// C11: s4_c11_0, c4_c11_0
// C12: s3_c12_0, c4_c12_0
// C13: s2_c13_0, c3_c13_0
// C14: pp[63], c2_c14_0
// ============================================================
  wire s5_c7_0;
  wire c5_c8_0;
  full_adder fa_s5_c7_0(
    .A(s3_c7_0), .B(c3_c7_0), .CarryIn(c4_c7_0),
    .Y(s5_c7_0), .CarryOut(c5_c8_0)
  );
  // ============================================================
// FINAL WALLACE COLUMN INVENTORY
// C0 : pp[0]
// C1 : pp[8], pp[1]
// C2 : s1_c2_0
// C3 : s2_c3_0
// C4 : s3_c4_0
// C5 : s4_c5_0
// C6 : s4_c6_0, c4_c6_0
// C7 : s5_c7_0
// C8 : s4_c8_0, c5_c8_0
// C9 : s4_c9_0, c4_c9_0
// C10: s4_c10_0, c4_c10_0
// C11: s4_c11_0, c4_c11_0
// C12: s3_c12_0, c4_c12_0
// C13: s2_c13_0, c3_c13_0
// C14: pp[63], c2_c14_0
// C15: none
// ============================================================
  assign sum = {1'b0, pp[63], s2_c13_0, s3_c12_0, s4_c11_0, s4_c10_0, s4_c9_0, s4_c8_0, s5_c7_0, s4_c6_0, s4_c5_0, s3_c4_0, s2_c3_0, s1_c2_0, pp[8], pp[0]};
  assign carry = {1'b0, c2_c14_0, c3_c13_0, c4_c12_0, c4_c11_0, c4_c10_0, c4_c9_0, c5_c8_0, 1'b0, c4_c6_0, 1'b0, 1'b0, 1'b0, 1'b0, pp[1], 1'b0};
endmodule
`default_nettype wire
