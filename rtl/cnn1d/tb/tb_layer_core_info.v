`timescale 1ns/1ps
// INFO (0x15) cua layer_core: phan cung tu khai bao cau hinh (mang PE, clock, baud, SRAM...).
// Chay o muc goi (rx_* / tx_*), khong can UART. Thu 2 cau hinh de chung minh gia tri
// do THAM SO synthesis quyet dinh (khong phai hang so trong host).
module tb_layer_core_info;
	localparam MP = 256;

	reg clk = 0, reset = 1;
	always #5 clk = ~clk;

	// ---- cau hinh A: 7x7 @ 50 MHz / 115200 ----
	reg [7:0] rx_cmd = 0, rx_seq = 0;
	reg [15:0] rx_length = 0;
	reg [MP*8-1:0] rx_payload = 0;
	reg rx_valid = 0, rx_error = 0;
	wire rx_enable, tx_start;
	wire [7:0] tx_cmd, tx_seq;
	wire [15:0] tx_length;
	wire [MP*8-1:0] tx_payload;

	layer_core #(.ARRAY_ROWS(7), .ARRAY_COLS(7), .CLK_FREQ(50000000), .BAUD_RATE(115200), .MAX_PAYLOAD(MP)) dut_a (
		.i_clk(clk), .i_reset(reset),
		.i_rx_cmd(rx_cmd), .i_rx_seq(rx_seq), .i_rx_length(rx_length), .i_rx_payload(rx_payload),
		.i_rx_valid(rx_valid), .i_rx_error(rx_error), .i_tx_busy(1'b0),
		.o_rx_enable(rx_enable), .o_tx_start(tx_start), .o_tx_cmd(tx_cmd), .o_tx_seq(tx_seq),
		.o_tx_length(tx_length), .o_tx_payload(tx_payload)
	);

	// ---- cau hinh B: 3x3 @ 25 MHz / 230400, SRAM nho hon ----
	reg [7:0] rx_cmd_b = 0, rx_seq_b = 0;
	reg [15:0] rx_length_b = 0;
	reg rx_valid_b = 0;
	wire rx_enable_b, tx_start_b;
	wire [7:0] tx_cmd_b, tx_seq_b;
	wire [15:0] tx_length_b;
	wire [MP*8-1:0] tx_payload_b;

	layer_core #(.ARRAY_ROWS(3), .ARRAY_COLS(3), .CLK_FREQ(25000000), .BAUD_RATE(230400), .MAX_PAYLOAD(MP),
		.IFMAP_MEM_DEPTH(2048), .WGT_MEM_DEPTH(8192), .FEEDER_DEPTH(512), .COUT_MAX(32)) dut_b (
		.i_clk(clk), .i_reset(reset),
		.i_rx_cmd(rx_cmd_b), .i_rx_seq(rx_seq_b), .i_rx_length(rx_length_b), .i_rx_payload({MP*8{1'b0}}),
		.i_rx_valid(rx_valid_b), .i_rx_error(1'b0), .i_tx_busy(1'b0),
		.o_rx_enable(rx_enable_b), .o_tx_start(tx_start_b), .o_tx_cmd(tx_cmd_b), .o_tx_seq(tx_seq_b),
		.o_tx_length(tx_length_b), .o_tx_payload(tx_payload_b)
	);

	integer errors = 0;
	reg got = 0;
	reg [7:0] cap_cmd, cap_seq;
	reg [15:0] cap_len;
	reg [MP*8-1:0] cap_pay;
	always @(posedge clk) if (tx_start) begin
		got <= 1; cap_cmd <= tx_cmd; cap_seq <= tx_seq; cap_len <= tx_length; cap_pay <= tx_payload;
	end

	task send_a;
		input [7:0] c;
		input [15:0] l;
		input [7:0] s;
		begin
			wait (rx_enable);
			@(negedge clk);
			rx_cmd = c; rx_seq = s; rx_length = l; rx_valid = 1; got = 0;
			@(negedge clk);
			rx_valid = 0;
			repeat (20) @(negedge clk);
			if (!got) begin errors = errors + 1; $display("[FAIL] no reply to cmd 0x%02x", c); end
		end
	endtask

	task check;
		input [8*32-1:0] what;
		input [31:0] a, b;
		begin
			if (a !== b) begin
				errors = errors + 1;
				$display("[FAIL] %0s: got=%0d want=%0d", what, a, b);
			end
		end
	endtask

	initial begin
		repeat (4) @(negedge clk);
		reset = 0;
		repeat (3) @(negedge clk);

		// A: INFO hop le
		send_a(8'h15, 0, 8'h5A);
		check("A cmd (RESULT=0x82)", cap_cmd, 8'h82);
		check("A seq", cap_seq, 8'h5A);
		check("A length", cap_len, 22);
		check("A version", cap_pay[7:0], 1);
		check("A rows", cap_pay[15:8], 7);
		check("A cols", cap_pay[23:16], 7);
		check("A reserved", cap_pay[31:24], 0);
		check("A clk_hz", cap_pay[63:32], 50000000);
		check("A baud", cap_pay[95:64], 115200);
		check("A ifmap depth", cap_pay[111:96], 4096);
		check("A weight depth", cap_pay[127:112], 16384);
		check("A feeder depth", cap_pay[143:128], 1024);
		check("A cout_max", cap_pay[159:144], 64);
		check("A max payload", cap_pay[175:160], MP);

		// A: INFO co payload -> NACK tham so sai
		send_a(8'h15, 1, 8'h01);
		check("A bad-len cmd (NACK=0x81)", cap_cmd, 8'h81);
		check("A bad-len code", cap_pay[15:8], 2);

		// A: lenh la van bi NACK 4 (INFO khong lam hong nhanh default)
		send_a(8'h17, 0, 8'h02);
		check("A unknown cmd", cap_cmd, 8'h81);
		check("A unknown code", cap_pay[15:8], 4);

		// B: cau hinh khac -> thong tin khac
		wait (rx_enable_b);
		@(negedge clk);
		rx_cmd_b = 8'h15; rx_seq_b = 8'h33; rx_length_b = 0; rx_valid_b = 1;
		@(negedge clk);
		rx_valid_b = 0;
		repeat (20) @(negedge clk);
		check("B length", tx_length_b, 22);
		check("B rows", tx_payload_b[15:8], 3);
		check("B cols", tx_payload_b[23:16], 3);
		check("B clk_hz", tx_payload_b[63:32], 25000000);
		check("B baud", tx_payload_b[95:64], 230400);
		check("B ifmap depth", tx_payload_b[111:96], 2048);
		check("B weight depth", tx_payload_b[127:112], 8192);
		check("B feeder depth", tx_payload_b[143:128], 512);
		check("B cout_max", tx_payload_b[159:144], 32);

		if (errors == 0) $display("PASS tb_layer_core_info: INFO reports 7x7 and 3x3 configurations correctly");
		else $display("[FAIL] tb_layer_core_info: %0d errors", errors);
		$finish;
	end

	initial begin
		#2000000;
		$display("[FAIL] TIMEOUT");
		$finish;
	end
endmodule
