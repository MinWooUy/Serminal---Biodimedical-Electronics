`timescale 1ns/1ps
module tb_crc16_stress;
    reg clk=0, reset=1, clear=0, valid=0, check=0;
    reg [7:0] data=0;
    reg [15:0] expected=16'hffff, received=0;
    wire [15:0] crc;
    wire ok, done;
    integer i;
    reg [31:0] random_state=32'h1bad_f00d;
    always #5 clk=~clk;

    crc_generator gen(clk, reset, clear, valid, data, crc);
    crc_checker chk(clk, reset, clear, valid, check, data, received, ok, done);

    function [15:0] reference_step;
        input [15:0] current;
        input [7:0] value;
        integer bit_number;
        reg [15:0] shifted;
        begin
            shifted=current ^ {value, 8'h00};
            for (bit_number=0; bit_number<8; bit_number=bit_number+1)
                if (shifted[15]) shifted=(shifted<<1)^16'h1021;
                else shifted=shifted<<1;
            reference_step=shifted;
        end
    endfunction

    task push_byte;
        input [7:0] value;
        begin
            @(negedge clk); data=value; valid=1;
            @(negedge clk); valid=0;
            expected=reference_step(expected, value);
        end
    endtask

    initial begin
        repeat(3) @(negedge clk); reset=0;
        for (i=0; i<1024; i=i+1) begin
            random_state={random_state[30:0], random_state[31]^random_state[21]^random_state[1]^random_state[0]};
            push_byte(random_state[7:0]);
            if ((i%17)==0) repeat(2) @(negedge clk);
        end
        if (crc !== expected) $fatal(1, "FAIL long stream CRC got %04h expected %04h", crc, expected);
        received=expected; check=1;
        @(negedge clk);
        if (!done || !ok) $fatal(1, "FAIL stress CRC valid compare");
        check=0; received=expected ^ 16'h0001; check=1;
        @(negedge clk);
        if (!done || ok) $fatal(1, "FAIL stress CRC corrupted compare");
        check=0; clear=1; expected=16'hffff;
        @(negedge clk); clear=0;
        if (crc !== 16'hffff || ok) $fatal(1, "FAIL stress CRC clear");
        push_byte(8'h00); push_byte(8'hff); push_byte(8'h55); push_byte(8'haa);
        if (crc !== expected) $fatal(1, "FAIL CRC after clear got %04h expected %04h", crc, expected);
        $display("PASS CRC 1024-byte stress, compare rejection, clear and restart");
        $finish;
    end

    initial begin #1000000; $fatal(1, "TIMEOUT CRC stress"); end
endmodule
