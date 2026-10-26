`timescale 1ns/1ps
module tb_post_process;
    reg clk=0, reset=1, valid=0, relu=0;
    always #5 clk=~clk;
    reg [287:0] tile=0;
    reg [23:0] shift=0;
    reg [1:0] rows=3, cols=3;
    wire [71:0] result;
    wire out_valid;

    conv1d_post_process dut(
        .i_clk(clk),.i_reset(reset),.i_valid(valid),.i_tile(tile),
        .i_shift(shift),.i_relu(relu),
        .i_active_rows(rows),.i_active_cols(cols),
        .o_tile(result),.o_valid(out_valid)
    );
    reg pv=0, clear=0;
    reg [23:0] pd=0;
    wire [23:0] po;
    wire pov;
    maxpool1d2 pool(clk,reset,clear,pv,pd,po,pov);
    integer fd,n,acc,s,act,want,i;
    reg [1023:0] path;
    initial begin
        if(!$value$plusargs("vectors=%s",path)) $fatal(1,"Missing vectors");
        fd=$fopen(path,"r");
        if(!fd) $fatal(1,"Cannot open vectors");
        #12; reset=0;
        while(!$feof(fd)) begin
            n=$fscanf(fd,"%d %d %d %d\n",acc,s,act,want);
            if(n!=4) $fatal(1,"Bad vector");
            @(negedge clk); valid=1; relu=act;
            for(i=0;i<9;i=i+1) tile[i*32 +: 32]=acc;
            for(i=0;i<3;i=i+1) begin shift[i*8 +: 8]=s; end
            @(posedge clk); #1;
            if(!out_valid) $fatal(1,"Missing valid");
            for(i=0;i<9;i=i+1)
                if($signed(result[i*8 +: 8])!==want)
                    $fatal(1,"Mismatch acc=%d shift=%d relu=%d actual=%d want=%d",acc,s,act,$signed(result[i*8 +: 8]),want);
        end
        // Distinct channel parameters, partial rows and columns.
        @(negedge clk); rows=2; cols=2; relu=1;
        tile={9{32'd20}}; shift={8'd0,8'd2,8'd1};
        @(posedge clk); #1;
        if(result!==72'h00000000050a00050a) $fatal(1,"Channel mapping/masking");
        @(negedge clk); valid=0;
        @(posedge clk); #1; if(out_valid) $fatal(1,"Valid bubble");
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
        #1; if(out_valid || pov || result!==0 || po!==0) $fatal(1,"Reset");
        $display("PASS post processing golden vectors, masking, bubbles, pooling, reset");
        $finish;
    end
    initial begin #1000000; $fatal(1,"Timeout"); end
endmodule
