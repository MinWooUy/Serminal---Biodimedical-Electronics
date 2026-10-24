`timescale 1ns/1ps
module tb_uart_rx_exhaustive;
    localparam DIV=16;
    reg clk=0, reset=1, rx=1;
    wire [7:0] data_out;
    wire ready, frame_error;
    integer received=0, i;
    always #5 clk=~clk;

    uart_receiver #(.CLK_FREQ(1600), .BAUD_RATE(100)) dut(
        .clk(clk), .reset(reset), .rx_in(rx), .data_out(data_out),
        .rx_ready(ready), .frame_error(frame_error));

    always @(negedge clk) begin
        if (!reset && ready) begin
            if (data_out !== received[7:0])
                $fatal(1, "FAIL RX byte %0d got %02h", received, data_out);
            received=received+1;
        end
        if (!reset && frame_error) $fatal(1, "FAIL unexpected framing error");
    end

    task send_byte;
        input [7:0] value;
        integer bit_number;
        begin
            @(negedge clk); rx=0; repeat(DIV) @(negedge clk);
            for (bit_number=0; bit_number<8; bit_number=bit_number+1) begin
                rx=value[bit_number]; repeat(DIV) @(negedge clk);
            end
            rx=1; repeat(DIV) @(negedge clk);
        end
    endtask

    initial begin
        repeat(4) @(negedge clk); reset=0;
        for (i=0; i<256; i=i+1) send_byte(i[7:0]);
        if (received != 256) $fatal(1, "FAIL exhaustive count %0d", received);

        // Reset during a partial frame; the following complete frame must work.
        @(negedge clk); rx=0; repeat(5) @(negedge clk);
        reset=1; rx=1; repeat(3) @(negedge clk); reset=0;
        received=8'h5a;
        send_byte(8'h5a);
        if (received != 8'h5b) $fatal(1, "FAIL receiver recovery after reset");
        $display("PASS UART RX exhaustive 256 bytes and reset recovery");
        $finish;
    end

    initial begin #10000000; $fatal(1, "TIMEOUT UART RX exhaustive"); end
endmodule
