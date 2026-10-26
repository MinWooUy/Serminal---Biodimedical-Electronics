// UART 8N1. A byte is accepted only when valid && ready.
module uart_transmitter #(
    parameter CLK_FREQ = 50000000,
    parameter BAUD_RATE = 115200
)(
    input wire clk, reset,
    input wire [7:0] data_in,
    input wire tx_start,
    output wire tx_ready,
    output wire tx_busy,
    output wire tx_out
);
    localparam DIV = CLK_FREQ / BAUD_RATE;
    localparam CW = (DIV <= 1) ? 1 : $clog2(DIV);
    reg [CW-1:0] count;
    reg [3:0] bit_index;
    reg [9:0] frame;
    reg busy;
    assign tx_ready = !busy;
    assign tx_busy = busy;
    assign tx_out = busy ? frame[0] : 1'b1;
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            count <= 0; bit_index <= 0; frame <= 10'h3ff; busy <= 0;
        end else if (!busy) begin
            if (tx_start) begin
                frame <= {1'b1, data_in, 1'b0};
                count <= 0; bit_index <= 0; busy <= 1;
            end
        end else if (count == DIV-1) begin
            count <= 0;
            if (bit_index == 9) busy <= 0;
            else begin
                frame <= {1'b1, frame[9:1]};
                bit_index <= bit_index + 1'b1;
            end
        end else count <= count + 1'b1;
    end
endmodule
