// Serial packet transport; no loopback and no accelerator side effects.
module uart_crc_top #(
    parameter CLK_FREQ=50000000, BAUD_RATE=115200, MAX_PAYLOAD=256,
    parameter RX_TIMEOUT_CLKS=CLK_FREQ/10
)(
    input wire clk, reset, uart_rx,
    output wire uart_tx,
    input wire rx_enable,
    output wire [7:0] rx_cmd, rx_seq,
    output wire [15:0] rx_length,
    output wire [MAX_PAYLOAD*8-1:0] rx_payload,
    output wire rx_valid, rx_error,
    input wire tx_start,
    input wire [7:0] tx_cmd, tx_seq,
    input wire [15:0] tx_length,
    input wire [MAX_PAYLOAD*8-1:0] tx_payload,
    output wire tx_busy
);
    wire [7:0] rd,td;
    wire rv,fe,tv,tr,pbusy,ubusy;
    uart_receiver #(.CLK_FREQ(CLK_FREQ),.BAUD_RATE(BAUD_RATE)) u_rx(
        .clk(clk),.reset(reset),.rx_in(uart_rx),.data_out(rd),
        .rx_ready(rv),.frame_error(fe));
    packet_rx #(.MAX_PAYLOAD(MAX_PAYLOAD),.TIMEOUT_CLKS(RX_TIMEOUT_CLKS)) u_prx(
        .clk(clk),.reset(reset),.enable(rx_enable),.byte_data(rd),
        .byte_valid(rv),.frame_error(fe),.cmd(rx_cmd),.seq(rx_seq),
        .length(rx_length),.payload(rx_payload),.packet_valid(rx_valid),.packet_error(rx_error));
    packet_tx #(.MAX_PAYLOAD(MAX_PAYLOAD)) u_ptx(
        .clk(clk),.reset(reset),.start(tx_start),.cmd(tx_cmd),.seq(tx_seq),
        .length(tx_length),.payload(tx_payload),.busy(pbusy),
        .byte_valid(tv),.byte_data(td),.byte_ready(tr));
    uart_transmitter #(.CLK_FREQ(CLK_FREQ),.BAUD_RATE(BAUD_RATE)) u_tx(
        .clk(clk),.reset(reset),.data_in(td),.tx_start(tv),
        .tx_ready(tr),.tx_busy(ubusy),.tx_out(uart_tx));
    assign tx_busy=pbusy|ubusy;
endmodule
