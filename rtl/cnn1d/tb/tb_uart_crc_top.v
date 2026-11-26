`timescale 1ns/1ps
// End-to-end physical UART stimulus and independent response decoder/CRC.
module tb_uart_crc_top;
    localparam DIV=16;
    reg clk=0,reset=1,rx=1;
    wire tx;
    always #5 clk=~clk;
    ecg_accel_top #(.CLK_FREQ(1600000),.BAUD_RATE(100000),
        .MEM_DEPTH(64),.RX_TIMEOUT_CLKS(800),.POST_ENABLE(1)) dut(clk,reset,rx,tx);
    reg [7:0] body[0:255];
    reg [7:0] reply[0:255];
    reg [7:0] b,rcmd,rseq,l0,l1,c0,c1;
    reg [15:0] crc;
    integer len,i,j,seqno=0,errors=0,writes=0,starts=0;
    integer expected[0:8];
    always @(posedge clk) begin
        if(dut.x_wr_en || dut.w_wr_en) writes=writes+1;
        if(dut.tile_start) starts=starts+1;
    end
    function [15:0] update;
        input [15:0] c;
        input [7:0] d;
        integer n;
        reg [15:0] t;
        begin
            t=c;
            for(n=7;n>=0;n=n-1)
                if(t[15]^d[n]) t=(t<<1)^16'h1021;
                else t=t<<1;
            update=t;
        end
    endfunction
    task send_byte;
        input [7:0] d;
        integer n;
        begin
            @(negedge clk);rx=0;
            repeat(DIV) @(negedge clk);
            for(n=0;n<8;n=n+1) begin
                rx=d[n];repeat(DIV) @(negedge clk);
            end
            rx=1;repeat(DIV) @(negedge clk);
        end
    endtask
    task get_byte;
        output [7:0] d;
        integer n;
        begin
            @(negedge tx);
            repeat(DIV/2) @(posedge clk);
            if(tx!==0) $fatal(1,"FAIL start bit");
            for(n=0;n<8;n=n+1) begin
                repeat(DIV) @(posedge clk);d[n]=tx;
            end
            repeat(DIV) @(posedge clk);
            if(tx!==1) $fatal(1,"FAIL stop bit");
        end
    endtask
    task send_packet;
        input [7:0] cmd;
        input integer size;
        input corrupt;
        reg [15:0] c;
        integer n;
        begin
            c=16'hffff;
            send_byte(8'ha5);send_byte(8'h5a);
            send_byte(cmd);c=update(c,cmd);
            send_byte(seqno);c=update(c,seqno);
            send_byte(size & 255);c=update(c,size & 255);
            send_byte(size >> 8);c=update(c,size >> 8);
            for(n=0;n<size;n=n+1) begin send_byte(body[n]);c=update(c,body[n]);end
            send_byte(c[7:0] ^ corrupt);send_byte(c[15:8]);
        end
    endtask
    task get_packet;
        input [7:0] wanted;
        reg [15:0] c;
        integer n;
        begin
            get_byte(b);if(b!==8'ha5) $fatal(1,"FAIL response SOF0");
            get_byte(b);if(b!==8'h5a) $fatal(1,"FAIL response SOF1");
            get_byte(rcmd);get_byte(rseq);get_byte(l0);get_byte(l1);
            len={l1,l0};if(len>256) $fatal(1,"FAIL length");
            c=update(16'hffff,rcmd);c=update(c,rseq);c=update(c,l0);c=update(c,l1);
            for(n=0;n<len;n=n+1) begin get_byte(reply[n]);c=update(c,reply[n]);end
            get_byte(c0);get_byte(c1);
            if({c1,c0}!==c || rcmd!==wanted || rseq!==seqno[7:0])
                $fatal(1,"FAIL response cmd=%h seq=%h CRC=%h expected=%h",rcmd,rseq,{c1,c0},c);
        end
    endtask
    task exchange;
        input [7:0] cmd;
        input integer size;
        input corrupt;
        input [7:0] wanted;
        begin
            fork
                send_packet(cmd,size,corrupt);
                get_packet(wanted);
            join
            repeat(DIV*2) @(negedge clk);
            seqno=(seqno+1)&255;
        end
    endtask
    task configure;
        input [7:0] rows,cols,kval;
        input [7:0] wanted;
        begin
            body[0]=kval;body[1]=0;body[2]=1;body[3]=0;
            body[4]=rows;body[5]=cols;body[6]=0;body[7]=0;body[8]=0;body[9]=0;
            exchange(3,10,0,wanted);
        end
    endtask
    task configure_preload;
        input [7:0] rows,cols,kval;
        input [7:0] wanted;
        begin
            body[0]=kval;body[1]=0;body[2]=1;body[3]=0;
            body[4]=rows;body[5]=cols;body[6]=0;body[7]=0;body[8]=0;body[9]=0;
            body[10]=1;
            exchange(3,11,0,wanted);
        end
    endtask
    task configure_post;
        input [7:0] wanted;
        begin
            // Per-column right shifts [1,2,0], ReLU enabled.
            body[0]=1;body[1]=2;body[2]=0;body[3]=1;
            exchange(7,4,0,wanted);
        end
    endtask
    task check_result;
        input integer rows,cols;
        reg signed [31:0] actual;
        integer n,post_actual,post_expected;
        begin
            if(len!=45) $fatal(1,"FAIL result length");
            for(n=0;n<9;n=n+1) begin
                actual={reply[n*4+3],reply[n*4+2],reply[n*4+1],reply[n*4]};
                if(n/3<rows && n%3<cols) begin
                    if(actual!==expected[n]) $fatal(1,"FAIL result %0d got %0d expected %0d",n,actual,expected[n]);
                end else if(actual!==0) $fatal(1,"FAIL inactive cell");
                post_expected=expected[n];
                case(n%3)
                    0: post_expected=(post_expected+1) >>> 1;
                    1: post_expected=(post_expected+2) >>> 2;
                    default: post_expected=post_expected;
                endcase
                if(post_expected<0) post_expected=0;
                if(post_expected>127) post_expected=127;
                if(n/3>=rows || n%3>=cols) post_expected=0;
                post_actual=$signed(reply[36+n]);
                if(post_actual!==post_expected)
                    $fatal(1,"FAIL post %0d got %0d expected %0d",n,post_actual,post_expected);
            end
        end
    endtask
    initial begin
        repeat(6) @(negedge clk);reset=0;
        repeat(6) @(negedge clk);
        // No data/config: RUN must not start.
        exchange(4,0,0,8'h81);
        if(starts!=0) $fatal(1,"FAIL unconfigured RUN");
        // Bad CRC must not cause even one memory write.
        body[0]=0;body[1]=0;body[2]=99;
        exchange(1,3,1,8'h81);
        if(writes!=0) $fatal(1,"FAIL CRC side effect");
        // Truncated packet times out and receiver can find next frame.
        send_byte(8'ha5);send_byte(8'h5a);send_byte(1);
        repeat(1000) @(negedge clk);
        exchange(5,0,0,8'h80);
        configure(3,3,17,8'h81);
        // Out-of-range write must not modify memory.
        body[0]=64;body[1]=0;body[2]=42;exchange(1,3,0,8'h81);
        if(writes!=0) $fatal(1,"FAIL address side effect");
        body[0]=0;body[1]=0;
        body[2]=1;body[3]=-2;body[4]=3;body[5]=4;body[6]=-1;
        exchange(1,7,0,8'h80);
        body[0]=0;body[1]=0;
        body[2]=2;body[3]=-1;body[4]=3;
        body[5]=1;body[6]=1;body[7]=1;
        body[8]=-1;body[9]=2;body[10]=0;
        exchange(2,11,0,8'h80);
        configure(3,3,3,8'h80);
        configure_post(8'h80);
        expected[0]=13;expected[1]=2;expected[2]=-5;
        expected[3]=5;expected[4]=5;expected[5]=8;
        expected[6]=-1;expected[7]=6;expected[8]=5;
        exchange(4,0,0,8'h82);check_result(3,3);
        // Bias uses the tile preload path. Reject RUN until every active column
        // has a valid INT32 bias, then broadcast each column bias to all rows.
        configure_preload(3,3,3,8'h80);
        exchange(4,0,0,8'h81);
        body[0]=0;body[1]=0;
        body[2]=10;body[3]=0;body[4]=0;body[5]=0;
        body[6]=-3;body[7]=-1;body[8]=-1;body[9]=-1;
        body[10]=7;body[11]=0;body[12]=0;body[13]=0;
        exchange(6,14,0,8'h80);
        expected[0]=23;expected[1]=-1;expected[2]=2;
        expected[3]=15;expected[4]=2;expected[5]=15;
        expected[6]=9;expected[7]=3;expected[8]=12;
        exchange(4,0,0,8'h82);check_result(3,3);
        // The legacy 10-byte configuration deliberately disables preload.
        configure(3,3,3,8'h80);
        expected[0]=13;expected[1]=2;expected[2]=-5;
        expected[3]=5;expected[4]=5;expected[5]=8;
        expected[6]=-1;expected[7]=6;expected[8]=5;
        // Reuse both read banks; no spurious swap.
        exchange(4,0,0,8'h82);check_result(3,3);
        // New ECG only: weight bank must be retained.
        body[0]=0;body[1]=0;
        body[2]=2;body[3]=2;body[4]=2;body[5]=2;body[6]=2;
        exchange(1,7,0,8'h80);
        for(i=0;i<9;i=i+1) begin
            case(i%3) 0:expected[i]=8;1:expected[i]=6;2:expected[i]=2;endcase
        end
        exchange(4,0,0,8'h82);check_result(3,3);
        configure(2,2,3,8'h80);
        exchange(4,0,0,8'h82);check_result(2,2);
        // Incomplete next staging bank cannot replace a complete active bank.
        body[0]=0;body[1]=0;body[2]=3;exchange(1,3,0,8'h80);
        j=starts;exchange(4,0,0,8'h81);
        if(starts!=j) $fatal(1,"FAIL incomplete staging bank");
        exchange(8'h7f,0,0,8'h81);
        // Fill a complete K=16 job in two chunks per tensor.
        expected[0]=0;
        for(i=0;i<16;i=i+1) expected[0]=expected[0]+(i-8)*((i%3)-1);
        for(j=0;j<2;j=j+1) begin
            body[0]=j*8;body[1]=0;
            for(i=0;i<8;i=i+1) body[i+2]=j*8+i-8;
            exchange(1,10,0,8'h80);
            body[0]=j*8;body[1]=0;
            for(i=0;i<8;i=i+1) body[i+2]=((j*8+i)%3)-1;
            exchange(2,10,0,8'h80);
        end
        configure(1,1,16,8'h80);
        exchange(4,0,0,8'h82);check_result(1,1);
        configure(1,1,1,8'h80);expected[0]=8;
        exchange(4,0,0,8'h82);check_result(1,1);
        $display("PASS UART -> CRC -> SRAM -> Conv1D -> UART integration");
        $finish;
    end
    initial begin #100000000; $fatal(1,"TIMEOUT integration");end
endmodule
