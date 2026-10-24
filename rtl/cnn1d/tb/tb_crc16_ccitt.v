`timescale 1ns/1ps
module tb_crc16_ccitt;
    reg clk=0,reset=1,clear=0,valid=0,check=0;
    reg [7:0] data=0;
    reg [15:0] received=0;
    wire [15:0] crc;
    wire ok,done;
    integer i;
    reg [71:0] text_data="123456789";
    initial begin
        $dumpfile("/tmp/tb_crc16_ccitt.vcd");
        $dumpvars(0, tb_crc16_ccitt);
    end
    always #5 clk=~clk;
    crc_generator gen(clk,reset,clear,valid,data,crc);
    crc_checker chk(clk,reset,clear,valid,check,data,received,ok,done);
    initial begin
        repeat(3) @(negedge clk);reset=0;
        for(i=0;i<9;i=i+1) begin
            data=text_data[(8-i)*8 +: 8];valid=1;
            @(negedge clk);valid=0;
            // Pauses must not accumulate the byte again.
            repeat(3) @(negedge clk);
        end
        if(crc!==16'h29b1) $fatal(1,"FAIL CRC known vector: %h",crc);
        received=16'h29b1;check=1;
        @(negedge clk);check=0;
        if(!done || !ok) $fatal(1,"FAIL CRC compare");
        @(negedge clk);
        if(done) $fatal(1,"FAIL check_done not pulse");
        received=16'h29b0;check=1;
        @(negedge clk);check=0;
        if(!done || ok) $fatal(1,"FAIL CRC corruption accepted");
        clear=1;@(negedge clk);clear=0;
        if(crc!==16'hffff || ok) $fatal(1,"FAIL CRC clear");
        // Compare including the final byte on the same edge.
        for(i=0;i<9;i=i+1) begin
            data=text_data[(8-i)*8 +: 8];valid=1;
            check=(i==8);received=16'h29b1;
            @(negedge clk);
        end
        valid=0;check=0;
        if(!done || !ok) $fatal(1,"FAIL same-cycle final-byte check");
        $display("PASS CRC known vector, gaps, clear, corruption, final-byte timing");
        $finish;
    end
    initial begin #100000; $fatal(1,"TIMEOUT CRC");end
endmodule
