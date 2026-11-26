`timescale 1ns/1ps
module tb_post_process;
    reg clk=0, reset=1, valid=0, relu=0;
    always #5 clk=~clk;
    reg [287:0] tile=0;
    reg [23:0] shift=0;
    reg [1:0] rows=3, cols=3;
    wire [71:0] result;
    wire out_valid, busy;

    conv1d_post_process dut(
        .i_clk(clk),.i_reset(reset),.i_valid(valid),.i_tile(tile),
        .i_shift(shift),.i_relu(relu),
        .i_active_rows(rows),.i_active_cols(cols),
        .o_tile(result),.o_valid(out_valid),.o_busy(busy)
    );

    // 7x7 instance, compared against the original 64-bit reference.
    reg valid7=0, relu7=0;
    reg [1567:0] tile7=0;
    reg [55:0] shift7=0;
    reg [2:0] rows7=7, cols7=7;
    wire [391:0] result7;
    wire ov7, busy7;
    conv1d_post_process #(.ROWS(7),.COLS(7)) dut7(
        .i_clk(clk),.i_reset(reset),.i_valid(valid7),.i_tile(tile7),
        .i_shift(shift7),.i_relu(relu7),
        .i_active_rows(rows7),.i_active_cols(cols7),
        .o_tile(result7),.o_valid(ov7),.o_busy(busy7)
    );

    reg pv=0, clear=0;
    reg [23:0] pd=0;
    wire [23:0] po;
    wire pov;
    maxpool1d2 pool(clk,reset,clear,pv,pd,po,pov);

    function signed [7:0] ref_q;
        input signed [31:0] acc;
        input signed [7:0] sh;
        input rl;
        reg signed [63:0] value;
        begin
            value = {{32{acc[31]}},acc};
            if (sh > 0) begin
                if (sh >= 34) value = 0;
                else value = (value + (64'sd1 <<< (sh-1))) >>> sh;
            end else if (sh < 0) begin
                if (sh <= -8) begin
                    if (value > 0) value = 127;
                    else if (value < 0) value = -127;
                end else value = value <<< (-sh);
            end
            if (rl && value < 0) value = 0;
            if (value > 127) value = 127;
            if (value < -127) value = -127;
            ref_q = value[7:0];
        end
    endfunction

    integer fd,n,acc,s,act,want,i,k,it,r,c;
    reg [1023:0] path;
    reg signed [31:0] a32;
    reg signed [7:0] s8;
    reg signed [7:0] exp8;
    integer seed;

    // Pulse i_valid for one cycle, then wait for o_valid after exactly
    // ROWS cycles (ROWS+1 cycles after the i_valid edge).
    task run3;
        begin
            @(negedge clk); valid=1;
            @(negedge clk); valid=0;
            if(!busy) $fatal(1,"Not busy after accept");
            for(k=0;k<2;k=k+1) begin
                @(posedge clk); #1;
                if(out_valid) $fatal(1,"Valid too early");
            end
            @(posedge clk); #1;
            if(!out_valid) $fatal(1,"Missing valid");
            if(busy) $fatal(1,"Busy with valid");
        end
    endtask

    initial begin
        if(!$value$plusargs("vectors=%s",path)) $fatal(1,"Missing vectors");
        fd=$fopen(path,"r");
        if(!fd) $fatal(1,"Cannot open vectors");
        #12; reset=0;
        while(!$feof(fd)) begin
            n=$fscanf(fd,"%d %d %d %d\n",acc,s,act,want);
            if(n!=4) $fatal(1,"Bad vector");
            relu=act;
            for(i=0;i<9;i=i+1) tile[i*32 +: 32]=acc;
            for(i=0;i<3;i=i+1) begin shift[i*8 +: 8]=s; end
            run3;
            for(i=0;i<9;i=i+1)
                if($signed(result[i*8 +: 8])!==want)
                    $fatal(1,"Mismatch acc=%d shift=%d relu=%d actual=%d want=%d",acc,s,act,$signed(result[i*8 +: 8]),want);
        end
        // Distinct channel parameters, partial rows and columns.
        rows=2; cols=2; relu=1;
        tile={9{32'd20}}; shift={8'd0,8'd2,8'd1};
        run3;
        if(result!==72'h00000000050a00050a) $fatal(1,"Channel mapping/masking");
        @(posedge clk); #1; if(out_valid) $fatal(1,"Valid bubble");

        // Different rows must reach their own lane: each element is distinct.
        rows=3; cols=3; relu=0; shift={8'd0,8'd0,8'd0};
        for(i=0;i<9;i=i+1) tile[i*32 +: 32]=i+1;
        run3;
        for(i=0;i<9;i=i+1)
            if($signed(result[i*8 +: 8])!==i+1) $fatal(1,"Row routing element %0d",i);

        // i_valid while busy is ignored; config is latched at accept.
        relu=0; rows=3; cols=3; shift={8'd0,8'd0,8'd0};
        for(i=0;i<9;i=i+1) tile[i*32 +: 32]=i+10;
        @(negedge clk); valid=1;
        @(negedge clk); valid=0; shift={8'd5,8'd5,8'd5}; rows=1; cols=1;
        @(posedge clk); #1;
        @(negedge clk); valid=1;
        @(negedge clk); valid=0;
        @(posedge clk); #1; if(!out_valid) $fatal(1,"Second accept not ignored");
        for(i=0;i<9;i=i+1)
            if($signed(result[i*8 +: 8])!==i+10) $fatal(1,"Latched config lost %0d",i);
        @(posedge clk); #1; if(out_valid || busy) $fatal(1,"Extra valid");

        // 7x7 randomized equivalence against the 64-bit reference.
        seed=1234;
        for(it=0;it<300;it=it+1) begin
            relu7=$random(seed);
            rows7=1+({$random(seed)}%7); cols7=1+({$random(seed)}%7);
            for(c=0;c<7;c=c+1) begin
                case({$random(seed)}%6)
                    0: s8 = -128 + ({$random(seed)}%256);
                    1: s8 = 0;
                    default: s8 = -9 + ({$random(seed)}%48);
                endcase
                shift7[c*8 +: 8] = s8;
            end
            for(i=0;i<49;i=i+1) begin
                case({$random(seed)}%4)
                    0: a32 = $random(seed);
                    1: a32 = $random(seed) >>> 16;
                    2: a32 = ($random(seed) % 1024);
                    default: a32 = ({$random(seed)}%2) ? 32'sh7fffffff : 32'sh80000000;
                endcase
                tile7[i*32 +: 32] = a32;
            end
            @(negedge clk); valid7=1;
            @(negedge clk); valid7=0;
            for(k=0;k<6;k=k+1) begin
                @(posedge clk); #1;
                if(ov7) $fatal(1,"7x7 valid too early");
            end
            @(posedge clk); #1;
            if(!ov7) $fatal(1,"7x7 missing valid");
            for(r=0;r<7;r=r+1)
                for(c=0;c<7;c=c+1) begin
                    if(r<rows7 && c<cols7)
                        exp8 = ref_q(tile7[(r*7+c)*32 +: 32], shift7[c*8 +: 8], relu7);
                    else exp8 = 0;
                    if($signed(result7[(r*7+c)*8 +: 8])!==exp8)
                        $fatal(1,"7x7 mismatch it=%0d r=%0d c=%0d acc=%0d sh=%0d relu=%0d got=%0d want=%0d",
                               it,r,c,$signed(tile7[(r*7+c)*32 +: 32]),$signed(shift7[c*8 +: 8]),relu7,
                               $signed(result7[(r*7+c)*8 +: 8]),exp8);
                end
        end

        // Pool across a bubble; signed comparisons on independent channels.
        @(negedge clk); pv=1; pd={8'd3,8'hfb,8'd7};
        @(posedge clk); #1; if(pov) $fatal(1,"Premature pool");
        @(negedge clk); pv=0;
        @(posedge clk); #1; if(pov) $fatal(1,"Pool bubble");
        @(negedge clk); pv=1; pd={8'd9,8'hfd,8'd2};
        @(posedge clk); #1; if(!pov || po!==24'h09fd07) $fatal(1,"Pool compare");
        @(negedge clk); pd=24'h010101;
        @(posedge clk); #1; if(pov) $fatal(1,"First of pair");
        @(negedge clk); clear=1;
        @(posedge clk); #1; if(pov) $fatal(1,"Clear priority");
        @(negedge clk); clear=0; pd=24'h020202;
        @(posedge clk); #1; if(pov) $fatal(1,"Odd tail not discarded");
        @(negedge clk); pd=24'h030303;
        @(posedge clk); #1; if(!pov || po!==24'h030303) $fatal(1,"Pool after clear");
        @(negedge clk); reset=1;
        #1; if(out_valid || busy || ov7 || pov || result!==0 || result7!==0 || po!==0) $fatal(1,"Reset");
        $display("PASS post processing golden vectors, 7x7 reference match, masking, latency, pooling, reset");
        $finish;
    end
    initial begin #20000000; $fatal(1,"Timeout"); end
endmodule
