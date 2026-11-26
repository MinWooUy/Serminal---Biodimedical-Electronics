`timescale 1ns/1ps
module tb_packet_rx;
    reg clk=0,reset=1,enable=1,valid=0,fe=0;
    reg [7:0] data=0;
    wire [7:0] cmd,seq;
    wire [15:0] length;
    wire [2047:0] payload;
    wire accepted,rejected;
    integer accepts=0,rejects=0,i;
    reg [15:0] crc;
    always #5 clk=~clk;
    packet_rx #(.TIMEOUT_CLKS(20)) dut(clk,reset,enable,data,valid,fe,
        cmd,seq,length,payload,accepted,rejected);
    always @(negedge clk) begin
        if(accepted) accepts=accepts+1;
        if(rejected) rejects=rejects+1;
    end
    function [15:0] step;
        input [15:0] c;
        input [7:0] d;
        reg [15:0] t;
        integer b;
        begin
            t=c;
            for(b=7;b>=0;b=b-1)
                if(t[15]^d[b]) t=(t<<1)^16'h1021;
                else t=t<<1;
            step=t;
        end
    endfunction
    task byte_in;
        input [7:0] value;
        begin @(negedge clk);data=value;valid=1;
            @(negedge clk);valid=0;end
    endtask
    task covered;
        input [7:0] value;
        begin byte_in(value);crc=step(crc,value);end
    endtask
    task header;
        input [15:0] n;
        begin
            byte_in(8'ha5);byte_in(8'h5a);crc=16'hffff;
            covered(1);covered(8'hff);covered(n[7:0]);covered(n[15:8]);
        end
    endtask
    initial begin
        repeat(4) @(negedge clk);reset=0;
        header(256);
        // All byte values include embedded A5/5A and must survive unchanged.
        for(i=0;i<256;i=i+1) covered(i);
        byte_in(crc[7:0]);byte_in(crc[15:8]);
        repeat(3) @(negedge clk);
        if(accepts!=1 || rejects!=0 || length!=256 || seq!=255 || cmd!=1)
            $fatal(1,"FAIL max packet");
        for(i=0;i<256;i=i+1)
            if(payload[i*8 +: 8]!==i[7:0]) $fatal(1,"FAIL payload byte %0d",i);
        header(257);repeat(3) @(negedge clk);
        if(rejects!=1 || accepts!=1) $fatal(1,"FAIL oversize rejected");
        header(0);byte_in(crc[7:0]^1);byte_in(crc[15:8]);
        repeat(3) @(negedge clk);
        if(rejects!=2 || accepts!=1) $fatal(1,"FAIL corrupt CRC");
        header(5);covered(42);repeat(25) @(negedge clk);
        header(0);byte_in(crc[7:0]);byte_in(crc[15:8]);
        repeat(3) @(negedge clk);
        if(accepts!=2) $fatal(1,"FAIL timeout recovery");
        header(1);@(negedge clk);fe=1;@(negedge clk);fe=0;
        header(0);byte_in(crc[7:0]);byte_in(crc[15:8]);
        repeat(3) @(negedge clk);
        if(accepts!=3) $fatal(1,"FAIL framing recovery");
        enable=0;
        header(0);byte_in(crc[7:0]);byte_in(crc[15:8]);
        repeat(3) @(negedge clk);
        if(accepts!=3) $fatal(1,"FAIL disabled parser");
        $display("PASS packet maximum length, byte transparency, oversized/CRC rejection, timeout, framing, disabled RX");
        $finish;
    end
    initial begin #100000; $fatal(1,"TIMEOUT packet");end
endmodule
