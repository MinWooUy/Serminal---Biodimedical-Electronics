`timescale 1ns/1ps

module tb_cnn_pipeline_controller;
    localparam LAYERS = 4;
    localparam CFG_W = 16;
    localparam ADDR_W = 16;
    localparam LAYER_W = 2;

    reg clk = 1'b0;
    reg reset = 1'b1;
    always #5 clk = ~clk;

    reg cfg_wr_en = 1'b0;
    reg [LAYER_W-1:0] cfg_wr_layer = 0;
    reg [CFG_W-1:0] cfg_wr_out_len = 0, cfg_wr_cout = 0, cfg_wr_k_len = 0;
    reg [CFG_W-1:0] cfg_wr_pad_next = 0;
    reg [ADDR_W-1:0] cfg_wr_step = 0;
    reg cfg_wr_pool = 0, cfg_wr_gap = 0, cfg_wr_valid = 0;
    reg run = 1'b0;
    reg layer_done = 1'b0;

    wire [LAYER_W-1:0] layer_index;
    wire layer_start, busy, done, error, input_bank, output_bank;
    wire [CFG_W-1:0] out_len, cout, k_len, pad_next;
    wire [ADDR_W-1:0] window_step;
    wire pool_en, gap_en, config_valid, all_config_valid;

    cnn_pipeline_controller #(.LAYERS(LAYERS)) dut (
        .i_clk(clk), .i_reset(reset),
        .i_cfg_wr_en(cfg_wr_en), .i_cfg_wr_layer(cfg_wr_layer),
        .i_cfg_wr_out_len(cfg_wr_out_len), .i_cfg_wr_cout(cfg_wr_cout),
        .i_cfg_wr_k_len(cfg_wr_k_len), .i_cfg_wr_window_step(cfg_wr_step),
        .i_cfg_wr_pool_en(cfg_wr_pool), .i_cfg_wr_gap_en(cfg_wr_gap),
        .i_cfg_wr_pad_next(cfg_wr_pad_next), .i_cfg_wr_valid(cfg_wr_valid),
        .i_run(run), .i_layer_done(layer_done),
        .o_layer_index(layer_index), .o_layer_start(layer_start),
        .o_busy(busy), .o_done(done), .o_error(error),
        .o_input_bank(input_bank), .o_output_bank(output_bank),
        .o_out_len(out_len), .o_cout(cout), .o_k_len(k_len),
        .o_window_step(window_step), .o_pool_en(pool_en), .o_gap_en(gap_en),
        .o_pad_next(pad_next), .o_config_valid(config_valid),
        .o_all_config_valid(all_config_valid)
    );

    integer errors = 0;
    integer starts = 0;
    integer i;
    reg [15:0] expected_out [0:3];
    reg [15:0] expected_cout [0:3];
    reg [15:0] expected_k [0:3];
    reg [15:0] expected_step [0:3];
    reg expected_pool [0:3];
    reg expected_gap [0:3];
    reg [15:0] expected_pad [0:3];

    task write_config;
        input integer n;
        begin
            @(negedge clk);
            cfg_wr_layer = n;
            cfg_wr_out_len = expected_out[n];
            cfg_wr_cout = expected_cout[n];
            cfg_wr_k_len = expected_k[n];
            cfg_wr_step = expected_step[n];
            cfg_wr_pool = expected_pool[n];
            cfg_wr_gap = expected_gap[n];
            cfg_wr_pad_next = expected_pad[n];
            cfg_wr_valid = 1'b1;
            cfg_wr_en = 1'b1;
            @(negedge clk);
            cfg_wr_en = 1'b0;
            cfg_wr_valid = 1'b0;
        end
    endtask

    task check_layer;
        input integer n;
        begin
            if (layer_index !== n || !config_valid || out_len !== expected_out[n] ||
                cout !== expected_cout[n] || k_len !== expected_k[n] ||
                window_step !== expected_step[n] || pool_en !== expected_pool[n] ||
                gap_en !== expected_gap[n] || pad_next !== expected_pad[n]) begin
                $display("[FAIL] layer %0d config/index mismatch", n);
                errors = errors + 1;
            end
        end
    endtask

    initial begin
        expected_out[0] = 360; expected_cout[0] = 16; expected_k[0] = 7;
        expected_step[0] = 1; expected_pool[0] = 1; expected_gap[0] = 0; expected_pad[0] = 2;
        expected_out[1] = 180; expected_cout[1] = 32; expected_k[1] = 80;
        expected_step[1] = 16; expected_pool[1] = 1; expected_gap[1] = 0; expected_pad[1] = 1;
        expected_out[2] = 90; expected_cout[2] = 64; expected_k[2] = 96;
        expected_step[2] = 32; expected_pool[2] = 1; expected_gap[2] = 0; expected_pad[2] = 1;
        expected_out[3] = 45; expected_cout[3] = 64; expected_k[3] = 192;
        expected_step[3] = 64; expected_pool[3] = 0; expected_gap[3] = 1; expected_pad[3] = 0;

        repeat (3) @(negedge clk);
        reset = 1'b0;
        repeat (2) @(negedge clk);

        // RUN khi chua du cau hinh phai bi tu choi.
        run = 1'b1;
        @(posedge clk);
        #1;
        run = 1'b0;
        if (!error || busy || done) begin
            $display("[FAIL] chap nhan chay khi cau hinh chua day du");
            errors = errors + 1;
        end

        for (i = 0; i < LAYERS; i = i + 1) write_config(i);
        @(negedge clk);
        if (!all_config_valid) begin
            $display("[FAIL] cac muc cau hinh chua hop le");
            errors = errors + 1;
        end

        run = 1'b1;
        @(posedge clk);
        #1;
        run = 1'b0;
        check_layer(0);
        if (!layer_start || input_bank !== 1'b0 || output_bank !== 1'b1) begin
            $display("[FAIL] start/bank cua layer 0");
            errors = errors + 1;
        end
        starts = starts + 1;

        for (i = 0; i < LAYERS; i = i + 1) begin
            @(posedge clk);
            #1;
            if (i == 0 && busy !== 1'b1) begin
                $display("[FAIL] layer 0 khong vao trang thai cho");
                errors = errors + 1;
            end
            @(negedge clk);
            layer_done = 1'b1;
            @(posedge clk);
            #1;
            layer_done = 1'b0;
            if (i < LAYERS - 1) begin
                check_layer(i + 1);
                if (!layer_start || input_bank !== ((i + 1) % 2) || output_bank !== (i % 2)) begin
                    $display("[FAIL] chuyen bank tai layer %0d", i + 1);
                    errors = errors + 1;
                end
                starts = starts + 1;
            end
        end

        @(posedge clk);
        #1;
        if (!done || busy || error || starts !== LAYERS) begin
            $display("[FAIL] trang thai cuoi done=%b busy=%b error=%b starts=%0d", done, busy, error, starts);
            errors = errors + 1;
        end
        if (errors == 0) $display("[PASS] luu cau hinh, start 4 layer va ping-pong bank");
        else $display("[FAIL] %0d loi", errors);
        $finish;
    end
endmodule
