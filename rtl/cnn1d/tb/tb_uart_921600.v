`timescale 1ns/1ps
// Physical timing check for the production 50 MHz / 921600 baud setting.
module tb_uart_921600;
    localparam CLK_FREQ = 50000000;
    localparam BAUD_RATE = 921600;
    localparam DIV = CLK_FREQ / BAUD_RATE;
    localparam BIT_NS = DIV * 20;

    reg clk = 0, reset = 1, start = 0;
    reg [7:0] data = 8'h55;
    wire tx, ready, busy;
    wire [7:0] received;
    wire valid, frame_error;
    reg rx = 1;
    integer tx_edges = 0;
    integer rx_count = 0;
    time previous_edge;

    always #10 clk = ~clk;

    uart_transmitter #(.CLK_FREQ(CLK_FREQ), .BAUD_RATE(BAUD_RATE)) tx_dut(
        .clk(clk), .reset(reset), .data_in(data), .tx_start(start),
        .tx_ready(ready), .tx_busy(busy), .tx_out(tx));
    uart_receiver #(.CLK_FREQ(CLK_FREQ), .BAUD_RATE(BAUD_RATE)) rx_dut(
        .clk(clk), .reset(reset), .rx_in(rx), .data_out(received),
        .rx_ready(valid), .frame_error(frame_error));

    always @(tx) begin
        if (!reset && busy) begin
            if (tx_edges != 0 && $time - previous_edge != BIT_NS)
                $fatal(1, "FAIL TX bit time %0t ns, expected %0t ns", $time - previous_edge, BIT_NS);
            previous_edge = $time;
            tx_edges = tx_edges + 1;
        end
    end

    always @(posedge valid) begin
        if (received !== 8'h3c || frame_error)
            $fatal(1, "FAIL RX at 921600: data=%02h error=%b", received, frame_error);
        rx_count = rx_count + 1;
    end

    task send_external_byte;
        input [7:0] value;
        integer bit_number;
        begin
            @(negedge clk); rx = 0; #(BIT_NS);
            for (bit_number = 0; bit_number < 8; bit_number = bit_number + 1) begin
                rx = value[bit_number]; #(BIT_NS);
            end
            rx = 1; #(BIT_NS);
        end
    endtask

    initial begin
        repeat (4) @(posedge clk);
        reset = 0;
        @(negedge clk); start = 1;
        wait (busy);
        wait (!busy);
        start = 0;
        if (tx_edges != 10) $fatal(1, "FAIL TX edge count %0d, expected 10", tx_edges);

        send_external_byte(8'h3c);
        #2000;
        if (rx_count != 1)
            $fatal(1, "FAIL RX event count at 921600: got %0d", rx_count);

        $display("PASS UART 921600 timing: DIV=%0d, bit=%0t ns", DIV, BIT_NS);
        $finish;
    end

    initial begin
        #1000000;
        $fatal(1, "TIMEOUT UART 921600");
    end
endmodule