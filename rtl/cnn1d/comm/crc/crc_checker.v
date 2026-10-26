module crc_checker(
    input wire clk, reset, clear, data_valid, check,
    input wire [7:0] data_in,
    input wire [15:0] crc_in,
    output reg crc_valid,
    output reg check_done
);
    reg [15:0] crc;
    wire [15:0] next_crc;
    crc16_ccitt_next u_next(crc, data_in, next_crc);
    always @(posedge clk or posedge reset) begin
        if (reset) begin crc <= 16'hffff; crc_valid <= 0; check_done <= 0; end
        else begin
            check_done <= 0;
            if (clear) begin crc <= 16'hffff; crc_valid <= 0; end
            else begin
                if (data_valid) crc <= next_crc;
                if (check) begin
                    crc_valid <= ((data_valid ? next_crc : crc) == crc_in);
                    check_done <= 1;
                end
            end
        end
    end
endmodule
