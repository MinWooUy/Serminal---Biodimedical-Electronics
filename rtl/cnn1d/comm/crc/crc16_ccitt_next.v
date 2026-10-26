// CRC-16/CCITT-FALSE byte update: poly 1021, no reflection, no xorout.
module crc16_ccitt_next(input wire [15:0] crc, input wire [7:0] data,
                        output reg [15:0] next_crc);
    integer i;
    always @* begin
        next_crc = crc ^ {data, 8'h00};
        for (i=0; i<8; i=i+1)
            if (next_crc[15]) next_crc = (next_crc << 1) ^ 16'h1021;
            else next_crc = next_crc << 1;
    end
endmodule
