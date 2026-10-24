`timescale 1ns/1ps
module tb_uart;
    localparam DIV=434;
    reg clk=0,reset=1,start=0;
    reg [7:0] data=0;
    wire tx,ready,busy,valid,err;
    wire [7:0] received;
    reg external_rx=1;
    wire ext_valid,ext_err;
    wire [7:0] ext_data;
    integer sent=0,got=0,n;
    integer ext_count=0,error_count=0;
    reg [7:0] ext_expected;
    initial begin
        $dumpfile("/tmp/tb_uart.vcd");
        $dumpvars(0, tb_uart);
    end
    always #10 clk=~clk;
    uart_transmitter tx_dut(.clk(clk),.reset(reset),.data_in(data),.tx_start(start),
        .tx_ready(ready),.tx_busy(busy),.tx_out(tx));
    uart_receiver rx_dut(.clk(clk),.reset(reset),.rx_in(tx),.data_out(received),
        .rx_ready(valid),.frame_error(err));
    uart_receiver ext_dut(.clk(clk),.reset(reset),.rx_in(external_rx),.data_out(ext_data),
        .rx_ready(ext_valid),.frame_error(ext_err));
    always @(negedge clk) begin
        if(!reset && valid) begin
            if(received!==got[7:0]) $fatal(1,"FAIL UART byte got %h expected %h",received,got[7:0]);
            got=got+1;
        end
        if(!reset && err) $fatal(1,"FAIL loopback framing");
        if(!reset && ext_valid) begin
            if(ext_data!==ext_expected) $fatal(1,"FAIL asynchronous data");
            ext_count=ext_count+1;
        end
        if(!reset && ext_err) error_count=error_count+1;
    end
    task external_byte;
        input [7:0] d;
        input integer period_ns;
        input good_stop;
        integer b;
        begin
            #7;external_rx=0;#(period_ns);
            for(b=0;b<8;b=b+1) begin external_rx=d[b];#(period_ns);end
            external_rx=good_stop;#(period_ns);
        end
    endtask
    initial begin
        repeat(4) @(posedge clk);#1;reset=0;
        // Keep valid asserted and replace data only on accepted transfers.
        @(negedge clk);start=1;data=0;
        while(sent<256) begin
            @(posedge clk);
            if(ready) begin
                sent=sent+1;
                @(negedge clk);
                data=sent[7:0];
                if(sent==256) start=0;
            end
        end
        wait(!busy);repeat(DIV) @(posedge clk);#1;
        if(got!=256 || tx!==1) $fatal(1,"FAIL UART count or idle");
        // Short start glitch must be rejected.
        external_rx=0;#1000;external_rx=1;#20000;
        if(ext_count || error_count) $fatal(1,"FAIL false start");
        ext_expected=8'ha5;external_byte(8'ha5,8500,1);#20000;
        ext_expected=8'h5a;external_byte(8'h5a,8850,1);#20000;
        if(ext_count!=2) $fatal(1,"FAIL baud tolerance");
        external_byte(0,8680,0);#50000;
        if(ext_count!=2 || error_count!=1) $fatal(1,"FAIL framing/break rejection");
        external_rx=1;#20000;
        ext_expected=8'hff;external_byte(8'hff,8680,1);#20000;
        if(ext_count!=3) $fatal(1,"FAIL framing recovery");
        $display("PASS UART 256 bytes, back-to-back, false start, baud mismatch, framing recovery");
        $finish;
    end
    initial begin #30000000; $fatal(1,"TIMEOUT UART");end
endmodule
