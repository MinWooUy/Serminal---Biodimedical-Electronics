`timescale 1ns/1ps

// PC-side UART frames through the real top, command decoder, ping-pong SRAM,
// Conv1D layer, and UART response path. Uses a 1-output, 1-channel, K=3 layer.
module tb_pc_to_fpga_layer_flow;
    localparam DIV = 16;

    reg clk = 1'b0;
    reg reset = 1'b1;
    reg uart_rx = 1'b1;
    wire uart_tx;
    always #5 clk = ~clk;

    ecg_accel_top #(
        .CLK_FREQ(1600000), .BAUD_RATE(100000), .MAX_PAYLOAD(64),
        .RX_TIMEOUT_CLKS(800), .IFMAP_MEM_DEPTH(64), .WGT_MEM_DEPTH(64),
        .FEEDER_DEPTH(16), .COUT_MAX(4)
    ) dut (.clk(clk), .reset(reset), .uart_rx(uart_rx), .uart_tx(uart_tx));

    reg [7:0] request_payload [0:255];
    reg [7:0] response_payload [0:255];
    reg [7:0] seqno = 0;
    reg [7:0] response_cmd, response_seq;
    reg [15:0] response_len;
    integer i;
    integer x_write_count = 0;
    integer w_write_count = 0;

    always @(posedge clk) begin
        if (dut.u_core.x_wr_en) x_write_count = x_write_count + 1;
        if (dut.u_core.w_wr_en) w_write_count = w_write_count + 1;
    end

    function [15:0] crc_step;
        input [15:0] crc_in;
        input [7:0] data_in;
        integer bit_no;
        reg [15:0] crc_work;
        begin
            crc_work = crc_in ^ {data_in, 8'h00};
            for (bit_no = 0; bit_no < 8; bit_no = bit_no + 1)
                crc_work = crc_work[15] ? ((crc_work << 1) ^ 16'h1021) : (crc_work << 1);
            crc_step = crc_work;
        end
    endfunction

    task send_uart_byte;
        input [7:0] data;
        integer bit_no;
        begin
            @(negedge clk); uart_rx = 1'b0;
            repeat (DIV) @(negedge clk);
            for (bit_no = 0; bit_no < 8; bit_no = bit_no + 1) begin
                uart_rx = data[bit_no];
                repeat (DIV) @(negedge clk);
            end
            uart_rx = 1'b1;
            repeat (DIV) @(negedge clk);
        end
    endtask

    task receive_uart_byte;
        output [7:0] data;
        integer bit_no;
        begin
            @(negedge uart_tx);
            repeat (DIV/2) @(posedge clk);
            if (uart_tx !== 1'b0) $fatal(1, "bad UART start bit");
            for (bit_no = 0; bit_no < 8; bit_no = bit_no + 1) begin
                repeat (DIV) @(posedge clk);
                data[bit_no] = uart_tx;
            end
            repeat (DIV) @(posedge clk);
            if (uart_tx !== 1'b1) $fatal(1, "bad UART stop bit");
        end
    endtask

    task send_frame;
        input [7:0] cmd;
        input integer payload_len;
        integer n;
        reg [15:0] crc;
        begin
            crc = 16'hffff;
            crc = crc_step(crc, cmd);
            crc = crc_step(crc, seqno);
            crc = crc_step(crc, payload_len[7:0]);
            crc = crc_step(crc, payload_len[15:8]);
            for (n = 0; n < payload_len; n = n + 1)
                crc = crc_step(crc, request_payload[n]);

            send_uart_byte(8'ha5);
            send_uart_byte(8'h5a);
            send_uart_byte(cmd);
            send_uart_byte(seqno);
            send_uart_byte(payload_len[7:0]);
            send_uart_byte(payload_len[15:8]);
            for (n = 0; n < payload_len; n = n + 1)
                send_uart_byte(request_payload[n]);
            send_uart_byte(crc[7:0]);
            send_uart_byte(crc[15:8]);
        end
    endtask

    task receive_frame;
        integer n;
        reg [7:0] b, len_lo, len_hi, crc_lo, crc_hi;
        reg [15:0] crc;
        begin
            receive_uart_byte(b);
            if (b !== 8'ha5) $fatal(1, "response SOF0=%02h", b);
            receive_uart_byte(b);
            if (b !== 8'h5a) $fatal(1, "response SOF1=%02h", b);
            receive_uart_byte(response_cmd);
            receive_uart_byte(response_seq);
            receive_uart_byte(len_lo);
            receive_uart_byte(len_hi);
            response_len = {len_hi, len_lo};
            if (response_len > 256) $fatal(1, "response too long: %0d", response_len);

            crc = 16'hffff;
            crc = crc_step(crc, response_cmd);
            crc = crc_step(crc, response_seq);
            crc = crc_step(crc, len_lo);
            crc = crc_step(crc, len_hi);
            for (n = 0; n < response_len; n = n + 1) begin
                receive_uart_byte(response_payload[n]);
                crc = crc_step(crc, response_payload[n]);
            end
            receive_uart_byte(crc_lo);
            receive_uart_byte(crc_hi);
            if ({crc_hi, crc_lo} !== crc)
                $fatal(1, "response CRC got=%04h expected=%04h", {crc_hi, crc_lo}, crc);
        end
    endtask

    task exchange;
        input [7:0] cmd;
        input integer payload_len;
        input [7:0] expected_response;
        input integer expected_len; // -1 skips length check
        input integer check_nack;
        input [7:0] expected_nack_code;
        begin
            // The FPGA can launch its reply as soon as the final request CRC
            // is accepted, so observe TX while the PC is still sending RX.
            fork
                send_frame(cmd, payload_len);
                receive_frame;
            join
            if (response_cmd !== expected_response)
                $fatal(1, "CMD %02h: response=%02h expected=%02h", cmd, response_cmd, expected_response);
            if (response_seq !== seqno)
                $fatal(1, "CMD %02h: response seq=%02h expected=%02h", cmd, response_seq, seqno);
            if (expected_len >= 0 && response_len !== expected_len)
                $fatal(1, "CMD %02h: response length=%0d expected=%0d", cmd, response_len, expected_len);
            if (check_nack && (response_payload[0] !== cmd || response_payload[1] !== expected_nack_code))
                $fatal(1, "CMD %02h: NACK payload=[%02h,%02h], expected code %02h",
                       cmd, response_payload[0], response_payload[1], expected_nack_code);
            $display("[CMD] 0x%02h -> 0x%02h, response length %0d", cmd, response_cmd, response_len);
            seqno = seqno + 1'b1;
        end
    endtask

    task write_x;
        begin
            request_payload[0] = 0; request_payload[1] = 0;
            request_payload[2] = 1; request_payload[3] = 2; request_payload[4] = 3;
            exchange(8'h08, 5, 8'h80, 2, 0, 0);
            if (dut.u_core.u_layer.u_tile.u_ifmap_mem.u_bank1.mem[0] !== 1 ||
                dut.u_core.u_layer.u_tile.u_ifmap_mem.u_bank1.mem[1] !== 2 ||
                dut.u_core.u_layer.u_tile.u_ifmap_mem.u_bank1.mem[2] !== 3)
                $fatal(1, "WRITE_X bytes are not in IFMap staging bank");
        end
    endtask

    task write_w;
        begin
            request_payload[0] = 0; request_payload[1] = 0;
            request_payload[2] = 4; request_payload[3] = 5; request_payload[4] = 6;
            exchange(8'h09, 5, 8'h80, 2, 0, 0);
            if (dut.u_core.u_layer.u_tile.u_wgt_mem.u_bank1.mem[0] !== 4 ||
                dut.u_core.u_layer.u_tile.u_wgt_mem.u_bank1.mem[1] !== 5 ||
                dut.u_core.u_layer.u_tile.u_wgt_mem.u_bank1.mem[2] !== 6)
                $fatal(1, "WRITE_W bytes are not in Weight staging bank");
        end
    endtask

    task configure_one_layer;
        begin
            // out_len=1, cout=1, K=3, step=1, pool=0, GAP=0, pad_next=0.
            request_payload[0]=1; request_payload[1]=0;
            request_payload[2]=1; request_payload[3]=0;
            request_payload[4]=3; request_payload[5]=0;
            request_payload[6]=1; request_payload[7]=0;
            request_payload[8]=0; request_payload[9]=0; request_payload[10]=0;
            exchange(8'h0d, 11, 8'h80, 2, 0, 0);
            if (dut.u_core.out_len !== 1 || dut.u_core.cout !== 1 || dut.u_core.k_len !== 3)
                $fatal(1, "CONFIG values did not reach layer core");
        end
    endtask

    integer cycles;
    initial begin
        repeat (6) @(negedge clk);
        reset = 1'b0;
        repeat (6) @(negedge clk);

        // Decoder status for RUN before configuration: NACK code 3.
        exchange(8'h0e, 0, 8'h81, 2, 1, 8'h03);

        // Invalid address must be rejected without a bank write.
        request_payload[0]=64; request_payload[1]=0; request_payload[2]=8'h7f;
        exchange(8'h08, 3, 8'h81, 2, 1, 8'h02);
        if (x_write_count != 0) $fatal(1, "invalid address caused an IFMap write");

        write_x;
        write_w;

        // CMD 0x0A: one signed INT32 bias, little endian (zero).
        request_payload[0]=0; request_payload[1]=0;
        request_payload[2]=0; request_payload[3]=0; request_payload[4]=0; request_payload[5]=0;
        exchange(8'h0a, 6, 8'h80, 2, 0, 0);
        if (dut.u_core.u_layer.bias_mem[0] !== 0) $fatal(1, "bias did not reach layer bank");

        // CMD 0x0B: shift for output channel 0.
        request_payload[0]=0; request_payload[1]=0; request_payload[2]=0;
        exchange(8'h0b, 3, 8'h80, 2, 0, 0);
        if (dut.u_core.u_layer.shift_mem[0] !== 0) $fatal(1, "shift did not reach layer bank");

        configure_one_layer;

        // CMD 0x0C flips IFMap and Weight staging banks into read ownership.
        request_payload[0]=3;
        exchange(8'h0c, 1, 8'h80, 2, 0, 0);
        if (dut.u_core.u_layer.u_tile.u_ifmap_mem.read_bank_q !== 1'b1 ||
            dut.u_core.u_layer.u_tile.u_wgt_mem.read_bank_q !== 1'b1)
            $fatal(1, "SWAP did not select the loaded banks");
        if (x_write_count != 3 || w_write_count != 3)
            $fatal(1, "unexpected write counts x=%0d w=%0d", x_write_count, w_write_count);

        // CMD 0x11..0x14 are reserved in this UART decoder and currently NACK unknown.
        exchange(8'h11, 0, 8'h81, 2, 1, 8'h04);
        exchange(8'h12, 0, 8'h81, 2, 1, 8'h04);
        exchange(8'h13, 0, 8'h81, 2, 1, 8'h04);
        request_payload[0]=0;
        exchange(8'h14, 1, 8'h81, 2, 1, 8'h04);

        // CMD 0x0E runs the Conv1D layer: 1*4 + 2*5 + 3*6 = 32.
        exchange(8'h0e, 0, 8'h82, 5, 0, 0);
        if (response_payload[0] !== 0) $fatal(1, "RUN reported layer error=%02h", response_payload[0]);
        cycles = {response_payload[4], response_payload[3], response_payload[2], response_payload[1]};
        if (cycles == 0) $fatal(1, "RUN returned zero cycles");
        if (dut.u_core.u_layer.u_tile.u_ifmap_mem.read_bank_q !== 1'b0)
            $fatal(1, "layer output bank did not become the read bank");

        // CMD 0x10 reads the post-processed output feature map from the new read bank.
        request_payload[0]=0; request_payload[1]=0; request_payload[2]=1;
        exchange(8'h10, 3, 8'h82, 1, 0, 0);
        if (response_payload[0] !== 8'd32)
            $fatal(1, "READ_FMAP got %0d, expected 32", $signed(response_payload[0]));

        // CMD 0x0F is the GAP result read path (zero because GAP was disabled).
        exchange(8'h0f, 0, 8'h82, 8, 0, 0);

        $display("[PASS] PC UART -> CMD decoder -> staging banks -> swap -> Conv1D layer -> output bank");
        $display("[INFO] RUN completed in %0d cycles; READ_FMAP[0]=32", cycles);
        $finish;
    end

    initial begin
        #100000000;
        $fatal(1, "testbench timeout");
    end
endmodule
