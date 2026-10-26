`timescale 1ns / 1ps

module tb_gvu_modules;
    localparam int LANES = 2;
    localparam int WIDTH = 8;
    localparam int ADDR_WIDTH = 4;

    logic clk = 1'b0;
    logic rst_n = 1'b0;
    always #5 clk = ~clk;

    logic rf_write_en;
    logic [1:0] rf_write_index, rf_read_index_a, rf_read_index_b;
    logic [LANES-1:0][WIDTH-1:0] rf_write_data, rf_read_data_a, rf_read_data_b;
    logic [LANES-1:0] rf_lane_mask;

    gvu_vector_register_file #(
        .LANES(LANES), .DATA_WIDTH(WIDTH), .REGISTER_COUNT(4), .REGISTER_INDEX_WIDTH(2)
    ) rf (
        .clk(clk), .rst_n(rst_n), .write_en(rf_write_en), .write_index(rf_write_index),
        .write_data(rf_write_data), .write_lane_mask(rf_lane_mask),
        .read_index_a(rf_read_index_a), .read_index_b(rf_read_index_b),
        .read_data_a(rf_read_data_a), .read_data_b(rf_read_data_b)
    );

    logic sp_write_en, sp_read_en, sp_read_valid;
    logic [2:0] sp_write_address, sp_read_address;
    logic [WIDTH-1:0] sp_write_data, sp_read_data;

    gvu_scratchpad_memory #(
        .DATA_WIDTH(WIDTH), .NUM_BANKS(2), .DEPTH(4), .ADDRESS_WIDTH(3)
    ) scratchpad (
        .clk(clk), .write_en(sp_write_en), .write_address(sp_write_address), .write_data(sp_write_data),
        .read_en(sp_read_en), .read_address(sp_read_address), .read_data(sp_read_data), .read_valid(sp_read_valid)
    );

    logic [LANES-1:0][WIDTH-1:0] stream_data;
    logic [LANES-1:0] stream_valid, stream_ready;
    logic [0:0] stream_select;
    logic [WIDTH-1:0] stream_out_data;
    logic stream_out_valid, stream_out_ready;

    gvu_stream_switch #(.LANES(LANES), .DATA_WIDTH(WIDTH), .SELECT_WIDTH(1)) stream_switch (
        .in_data(stream_data), .in_valid(stream_valid), .in_ready(stream_ready), .select(stream_select),
        .out_data(stream_out_data), .out_valid(stream_out_valid), .out_ready(stream_out_ready)
    );

    logic [LANES-1:0][WIDTH-1:0] psum_in;
    logic [LANES-1:0][15:0] accumulation, psum_out;
    logic accumulate_en;
    gvu_psum_if #(.LANES(LANES), .DATA_WIDTH(WIDTH), .ACC_WIDTH(16)) psum (
        .partial_sum_in(psum_in), .accumulation(accumulation), .accumulate_en(accumulate_en), .partial_sum_out(psum_out)
    );

    logic load_valid, load_ready, store_valid, store_ready;
    logic [ADDR_WIDTH-1:0] load_address, store_address;
    logic [WIDTH-1:0] load_data, store_data;
    logic mem_write_en, mem_read_en;
    logic [ADDR_WIDTH-1:0] mem_write_address, mem_read_address;
    logic [WIDTH-1:0] mem_write_data;
    gvu_vlsu #(.DATA_WIDTH(WIDTH), .ADDRESS_WIDTH(ADDR_WIDTH)) vlsu (
        .clk(clk), .rst_n(rst_n), .load_valid(load_valid), .load_ready(load_ready),
        .load_address(load_address), .load_data(load_data), .mem_write_en(mem_write_en),
        .mem_write_address(mem_write_address), .mem_write_data(mem_write_data),
        .store_valid(store_valid), .store_ready(store_ready), .store_address(store_address),
        .store_data(store_data), .mem_read_en(mem_read_en), .mem_read_address(mem_read_address)
    );

    logic dma_request_valid, dma_request_ready, dma_request_write;
    logic [ADDR_WIDTH-1:0] dma_request_address;
    logic [WIDTH-1:0] dma_request_wdata, dma_response_rdata, dma_rdata;
    logic dma_response_valid, dma_response_ready, dma_valid, dma_ready, dma_write;
    logic [ADDR_WIDTH-1:0] dma_address;
    logic [WIDTH-1:0] dma_wdata;
    gvu_dma_if #(.DATA_WIDTH(WIDTH), .ADDRESS_WIDTH(ADDR_WIDTH)) dma (
        .clk(clk), .rst_n(rst_n), .request_valid(dma_request_valid), .request_ready(dma_request_ready),
        .request_write(dma_request_write), .request_address(dma_request_address), .request_wdata(dma_request_wdata),
        .response_valid(dma_response_valid), .response_ready(dma_response_ready), .response_rdata(dma_response_rdata),
        .dma_valid(dma_valid), .dma_ready(dma_ready), .dma_write(dma_write), .dma_address(dma_address),
        .dma_wdata(dma_wdata), .dma_rdata(dma_rdata)
    );

    initial begin : tests
        rf_write_en = 1'b0;
        rf_write_index = '0;
        rf_read_index_a = '0;
        rf_read_index_b = '0;
        rf_write_data = '0;
        rf_lane_mask = '0;
        sp_write_en = 1'b0;
        sp_read_en = 1'b0;
        sp_write_address = '0;
        sp_read_address = '0;
        sp_write_data = '0;
        stream_data = '0;
        stream_valid = '0;
        stream_select = '0;
        stream_out_ready = 1'b1;
        psum_in = '0;
        accumulation = '0;
        accumulate_en = 1'b0;
        load_valid = 1'b0;
        store_valid = 1'b0;
        load_address = '0;
        store_address = '0;
        load_data = '0;
        store_data = '0;
        dma_request_valid = 1'b0;
        dma_request_write = 1'b0;
        dma_request_address = '0;
        dma_request_wdata = '0;
        dma_response_ready = 1'b0;
        dma_ready = 1'b1;
        dma_rdata = 8'h5a;

        repeat (2) @(negedge clk);
        rst_n = 1'b1;

        // Register file write with lane mask: lane 1 must remain zero.
        rf_write_index = 2'd1;
        rf_write_data[0] = 8'd11;
        rf_write_data[1] = 8'd22;
        rf_lane_mask = 2'b01;
        rf_write_en = 1'b1;
        @(posedge clk);
        #1;
        rf_write_en = 1'b0;
        rf_read_index_a = 2'd1;
        #1;
        if (rf_read_data_a[0] !== 8'd11 || rf_read_data_a[1] !== 8'd0) $fatal(1, "register file lane mask failed");

        // Banked scratchpad address 5 maps to bank 1, element 1.
        sp_write_address = 4'd5;
        sp_write_data = 8'haa;
        sp_write_en = 1'b1;
        @(posedge clk);
        #1;
        sp_write_en = 1'b0;
        sp_read_address = 4'd5;
        sp_read_en = 1'b1;
        @(posedge clk);
        #1;
        sp_read_en = 1'b0;
        if (!sp_read_valid || sp_read_data !== 8'haa) $fatal(1, "scratchpad banking failed");

        stream_data[0] = 8'h12;
        stream_data[1] = 8'h34;
        stream_valid = 2'b11;
        stream_select = 1'b1;
        #1;
        if (!stream_out_valid || stream_out_data !== 8'h34 || stream_ready !== 2'b10) $fatal(1, "stream switch failed");
        stream_out_ready = 1'b0;
        #1;
        if (stream_ready !== 2'b00) $fatal(1, "stream backpressure failed");
        stream_out_ready = 1'b1;

        psum_in[0] = 8'd3;
        psum_in[1] = 8'd4;
        accumulation[0] = 16'd10;
        accumulation[1] = 16'd20;
        accumulate_en = 1'b1;
        #1;
        if (psum_out[0] !== 16'd13 || psum_out[1] !== 16'd24) $fatal(1, "psum accumulate failed");

        load_valid = 1'b1;
        load_address = 4'd3;
        load_data = 8'h66;
        #1;
        if (!load_ready || !mem_write_en || mem_write_address !== 4'd3 || mem_write_data !== 8'h66) $fatal(1, "VLSU load failed");
        load_valid = 1'b0;
        store_valid = 1'b1;
        store_address = 4'd3;
        #1;
        if (!store_ready || !mem_read_en || mem_read_address !== 4'd3) $fatal(1, "VLSU store failed");
        store_valid = 1'b0;

        dma_request_valid = 1'b1;
        dma_request_write = 1'b0;
        dma_request_address = 4'd2;
        #1;
        if (!dma_valid || dma_address !== 4'd2) $fatal(1, "DMA request failed");
        @(posedge clk);
        #1;
        dma_request_valid = 1'b0;
        if (!dma_response_valid || dma_response_rdata !== 8'h5a) $fatal(1, "DMA read failed");
        dma_response_ready = 1'b1;
        @(posedge clk);
        #1;
        if (dma_response_valid) $fatal(1, "DMA response did not clear");

        $display("PASS independent GVU module tests");
        $finish;
    end
endmodule
