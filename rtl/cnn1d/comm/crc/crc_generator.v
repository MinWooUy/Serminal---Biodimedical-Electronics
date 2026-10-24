module crc_generator(
    input wire clk, reset, clear, data_valid,
    input wire [7:0] data_in,
    output reg [15:0] crc_out
);
    wire [15:0] next_crc;
    crc16_ccitt_next u_next(crc_out, data_in, next_crc);
    // clear has priority; send the first byte on a later cycle.
    always @(posedge clk or posedge reset)
        if (reset) crc_out <= 16'hffff;
        else if (clear) crc_out <= 16'hffff;
        else if (data_valid) crc_out <= next_crc;
endmodule
