`timescale 1ns/1ps
// System-level test: UART frames in, RESULT frame out, with POST_ENABLE=1.
// Sends the same command sequence as host/ecg_host.py for one 7x7 tile
// (K=7, rows=7, cols=7, preload bias, CONFIG_POST) and checks the RESULT
// payload (raw INT32 then post INT8) against a reference computed here.
module tb_ecg_accel_top #(parameter ARRAY_ROWS=7, ARRAY_COLS=7);
    localparam CLK_FREQ = 50000000;
    localparam BAUD     = 5000000;           // 10 clocks per bit keeps the test short
    localparam BIT_NS   = 1000000000/BAUD;
    localparam K_LEN    = 7;
    localparam N        = ARRAY_ROWS*ARRAY_COLS;
    localparam RAW_B    = N*4;
    localparam POST_B   = N;
    localparam SHIFT    = 1;

    reg clk = 0, reset = 1, uart_rx = 1;
    wire uart_tx;
    always #10 clk = ~clk;

    ecg_accel_top #(.CLK_FREQ(CLK_FREQ), .BAUD_RATE(BAUD), .POST_ENABLE(1),
                    .ARRAY_ROWS(ARRAY_ROWS), .ARRAY_COLS(ARRAY_COLS),
                    .FEEDER_DEPTH(1024)) dut(
        .clk(clk), .reset(reset), .uart_rx(uart_rx), .uart_tx(uart_tx));

    integer errors = 0;

    // ---- host side -------------------------------------------------------
    function [15:0] crc_step;
        input [15:0] crc; input [7:0] data;
        integer b; reg [15:0] c;
        begin
            c = crc ^ {data, 8'h00};
            for (b = 0; b < 8; b = b + 1)
                c = c[15] ? ((c << 1) ^ 16'h1021) : (c << 1);
            crc_step = c;
        end
    endfunction

    task send_byte; input [7:0] d; integer i;
        begin
            uart_rx = 0; #(BIT_NS);
            for (i = 0; i < 8; i = i + 1) begin uart_rx = d[i]; #(BIT_NS); end
            uart_rx = 1; #(BIT_NS);
        end
    endtask

    reg [7:0] tx_buf [0:299];
    reg [7:0] seq = 0;
    task send_frame; input [7:0] cmd; input integer len;
        integer i; reg [15:0] crc;
        begin
            crc = 16'hFFFF;
            crc = crc_step(crc, cmd); crc = crc_step(crc, seq);
            crc = crc_step(crc, len[7:0]); crc = crc_step(crc, len[15:8]);
            for (i = 0; i < len; i = i + 1) crc = crc_step(crc, tx_buf[i]);
            send_byte(8'hA5); send_byte(8'h5A);
            send_byte(cmd); send_byte(seq); send_byte(len[7:0]); send_byte(len[15:8]);
            for (i = 0; i < len; i = i + 1) send_byte(tx_buf[i]);
            send_byte(crc[7:0]); send_byte(crc[15:8]);   // CRC is little endian on the wire
        end
    endtask

    // ---- UART monitor ------------------------------------------------------
    reg [7:0] rx_buf [0:1023];
    integer rx_n = 0;
    reg [7:0] mon;
    integer mi;
    initial begin
        forever begin
            @(negedge uart_tx);
            #(BIT_NS/2);
            #(BIT_NS);
            for (mi = 0; mi < 8; mi = mi + 1) begin mon[mi] = uart_tx; #(BIT_NS); end
            rx_buf[rx_n] = mon; rx_n = rx_n + 1;
        end
    end

    // Wait for one whole response frame; returns its command in rsp_cmd.
    reg [7:0] rsp_cmd;
    integer rsp_len;
    task get_response; input integer timeout_us; integer waited, base;
        begin
            base = rx_n; waited = 0;
            while (!(rx_n - base >= 6 && rx_n - base >= 8 + {rx_buf[base+5], rx_buf[base+4]})
                   && waited < timeout_us) begin #1000; waited = waited + 1; end
            if (rx_n - base < 6 || rx_n - base < 8 + {rx_buf[base+5], rx_buf[base+4]}) begin
                $display("[FAIL] t=%0t TIMEOUT: no complete response (got %0d bytes)", $time, rx_n - base);
                errors = errors + 1; rsp_cmd = 0; rsp_len = 0;
            end else begin
                rsp_cmd = rx_buf[base+2];
                rsp_len = {rx_buf[base+5], rx_buf[base+4]};
                if (rx_buf[base+3] !== seq) begin
                    $display("[FAIL] t=%0t sequence mismatch got %0d want %0d", $time, rx_buf[base+3], seq);
                    errors = errors + 1;
                end
                // move the payload to the front so callers can read rx_buf[6+i]
            end
            seq = seq + 1;
        end
    endtask

    task expect_ack; input [7:0] cmd; input [8*16-1:0] name;
        begin
            get_response(200);
            if (rsp_cmd !== 8'h80) begin
                $display("[FAIL] t=%0t %0s: response 0x%02x (want ACK)", $time, name, rsp_cmd);
                errors = errors + 1;
            end else $display("[PASS] %0s ACK", name);
        end
    endtask

    // ---- reference ---------------------------------------------------------
    function integer sample_value; input integer a;
        begin sample_value = ((a * 5 + 3) % 17) - 8; end
    endfunction
    function integer weight_value; input integer c; input integer t;
        begin weight_value = ((c * 3 + t * 7 + 1) % 13) - 6; end
    endfunction
    function integer bias_value; input integer c;
        begin bias_value = 10 * c - 25; end
    endfunction

    integer i, r, c, t, acc, got, want_raw, want_post, base;
    reg [7:0] b8;
    reg signed [31:0] raw32;
    reg signed [7:0] p8;

    initial begin
        #100; reset = 0; #1000;

        // WRITE_IFMAP: address 0, ROWS+K-1 samples
        tx_buf[0] = 0; tx_buf[1] = 0;
        for (i = 0; i < ARRAY_ROWS + K_LEN - 1; i = i + 1) tx_buf[2+i] = sample_value(i);
        send_frame(8'h01, 2 + ARRAY_ROWS + K_LEN - 1); expect_ack(8'h01, "WRITE_IFMAP");

        // WRITE_WEIGHT: column c at address c*K
        tx_buf[0] = 0; tx_buf[1] = 0;
        for (c = 0; c < ARRAY_COLS; c = c + 1)
            for (t = 0; t < K_LEN; t = t + 1) tx_buf[2 + c*K_LEN + t] = weight_value(c, t);
        send_frame(8'h02, 2 + ARRAY_COLS*K_LEN); expect_ack(8'h02, "WRITE_WEIGHT");

        // WRITE_BIAS: INT32 little endian, one per column
        tx_buf[0] = 0; tx_buf[1] = 0;
        for (c = 0; c < ARRAY_COLS; c = c + 1) begin
            tx_buf[2+4*c]   = bias_value(c);
            tx_buf[3+4*c]   = bias_value(c) >>> 8;
            tx_buf[4+4*c]   = bias_value(c) >>> 16;
            tx_buf[5+4*c]   = bias_value(c) >>> 24;
        end
        send_frame(8'h06, 2 + 4*ARRAY_COLS); expect_ack(8'h06, "WRITE_BIAS");

        // CONFIG_POST: one shift per column then ReLU
        for (c = 0; c < ARRAY_COLS; c = c + 1) tx_buf[c] = SHIFT;
        tx_buf[ARRAY_COLS] = 1;
        send_frame(8'h07, ARRAY_COLS + 1); expect_ack(8'h07, "CONFIG_POST");

        // CONFIG_TILE: k, step, rows, cols, x_base, w_base, preload
        tx_buf[0] = K_LEN; tx_buf[1] = 0; tx_buf[2] = 1; tx_buf[3] = 0;
        tx_buf[4] = ARRAY_ROWS; tx_buf[5] = ARRAY_COLS;
        tx_buf[6] = 0; tx_buf[7] = 0; tx_buf[8] = 0; tx_buf[9] = 0; tx_buf[10] = 1;
        send_frame(8'h03, 11); expect_ack(8'h03, "CONFIG_TILE");

        // RUN_TILE
        send_frame(8'h04, 0);
        base = rx_n;
        get_response(1500);
        if (rsp_cmd !== 8'h82) begin
            $display("[FAIL] RUN_TILE: response 0x%02x (want RESULT 0x82)", rsp_cmd);
            errors = errors + 1;
        end else if (rsp_len !== RAW_B + POST_B) begin
            $display("[FAIL] RUN_TILE: payload %0d bytes (want %0d)", rsp_len, RAW_B + POST_B);
            errors = errors + 1;
        end else begin
            for (r = 0; r < ARRAY_ROWS; r = r + 1)
                for (c = 0; c < ARRAY_COLS; c = c + 1) begin
                    acc = bias_value(c);
                    for (t = 0; t < K_LEN; t = t + 1)
                        acc = acc + sample_value(r + t) * weight_value(c, t);
                    raw32 = {rx_buf[base+6+(r*ARRAY_COLS+c)*4+3], rx_buf[base+6+(r*ARRAY_COLS+c)*4+2],
                             rx_buf[base+6+(r*ARRAY_COLS+c)*4+1], rx_buf[base+6+(r*ARRAY_COLS+c)*4]};
                    p8 = rx_buf[base+6+RAW_B+r*ARRAY_COLS+c];
                    want_post = (acc + 1) >>> 1;
                    if (want_post < 0) want_post = 0;
                    if (want_post > 127) want_post = 127;
                    if (raw32 !== acc || p8 !== want_post) begin
                        $display("[FAIL] C[%0d][%0d] raw=%0d want=%0d post=%0d want=%0d",
                                 r, c, raw32, acc, p8, want_post);
                        errors = errors + 1;
                    end
                end
            if (errors == 0) $display("[PASS] RUN_TILE RESULT raw + post match (%0dx%0d)", ARRAY_ROWS, ARRAY_COLS);
        end

        if (errors == 0) $display("RESULT : ALL TESTS PASSED");
        else $display("RESULT : FAILED (%0d errors)", errors);
        $finish;
    end

    initial begin #50000000; $display("TIMEOUT"); $finish; end
endmodule
