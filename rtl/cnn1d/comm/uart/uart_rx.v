// UART 8N1, centre sampling. CLK_FREQ/BAUD_RATE must be >= 8.
module uart_receiver #(
    parameter CLK_FREQ = 50000000,
    parameter BAUD_RATE = 115200
)(
    input wire clk, reset, rx_in,
    output reg [7:0] data_out,
    output reg rx_ready,
    output reg frame_error
);
    localparam DIV = CLK_FREQ / BAUD_RATE;
    localparam CW = (DIV <= 1) ? 1 : $clog2(DIV);
    localparam IDLE=0, START=1, DATA=2, STOP=3, RECOVER=4;
    (* ASYNC_REG = "TRUE" *) reg rx_meta, rx_sync;
    reg [2:0] state, bit_index;
    reg [CW-1:0] count;
    reg [7:0] data;
    always @(posedge clk or posedge reset) begin
        if (reset) begin rx_meta <= 1; rx_sync <= 1; end
        else begin rx_meta <= rx_in; rx_sync <= rx_meta; end
    end
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            state <= IDLE; count <= 0; bit_index <= 0;
            data <= 0; data_out <= 0; rx_ready <= 0; frame_error <= 0;
        end else begin
            rx_ready <= 0; frame_error <= 0;
            case (state)
                IDLE: begin
                    count <= 0;
                    if (!rx_sync) state <= START;
                end
                START: begin
                    if (rx_sync) begin state <= IDLE; count <= 0; end
                    else if (count == DIV/2-1) begin
                        count <= 0; bit_index <= 0; state <= DATA;
                    end else count <= count + 1'b1;
                end
                DATA: begin
                    if (count == DIV-1) begin
                        count <= 0; data[bit_index] <= rx_sync;
                        if (bit_index == 7) state <= STOP;
                        else bit_index <= bit_index + 1'b1;
                    end else count <= count + 1'b1;
                end
                STOP: begin
                    if (count == DIV-1) begin
                        count <= 0;
                        if (rx_sync) begin
                            data_out <= data; rx_ready <= 1; state <= IDLE;
                        end else begin frame_error <= 1; state <= RECOVER; end
                    end else count <= count + 1'b1;
                end
                RECOVER: if (rx_sync) state <= IDLE;
                default: state <= IDLE;
            endcase
        end
    end
endmodule
