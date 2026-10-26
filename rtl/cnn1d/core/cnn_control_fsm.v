`default_nettype none
// start_tile + k_len
//
// Feeder tự:
// - rd_fire_all
// - k_cnt
// - first
// - last

//
// Controller
//			|
// 		Feeder_start --> IFMAP, Weight
//			|
// 		psm_clr --> PSM
// 		|
// 		|
// 		<--sa_tile_done -- SA
//			|
// 		scan_en --> SA
// 		capture_en --> PSM
// 		|
// 		<--capture_done --> PSM
// 		|
// 		|
//			Done

// Notice
// S_IDLE
//  |
//  | i_start = 1
//  v
//S_LOAD
//  | o_feeder_start = 1 pulse
//  | o_psm_clr      = 1 pulse
//  v
//S_COMPUTE
//  |
//  | i_sa_tile_done = 1
//  v
//S_DRAIN
//  | o_scan_en        = 1
//  | o_psm_capture_en = 1
//  |
//  | i_psm_capture_done = 1
//  v
//S_DONE
//  | done = 1
//  |
//  | i_start = 0
//  v
//S_IDLE

module cnn_control_fsm (
	input wire i_clk,
	input wire i_reset,
	input wire i_start,
	
	// Completion
	input wire i_sa_tile_done,
	input wire i_psm_capture_done,
	input wire i_preload_done,
	
	// Feeder
	output reg o_feeder_start,
	
	//SA
	output reg o_scan_en,
	
	// PSM
	output reg o_psm_clr,
	output reg o_psm_capture_en,
	// PSM tự đếm sô lân dịch
	output reg o_preload_start,
	
	// status
	output reg busy,
	output reg done,
	
	output wire[2:0] state_dbg
);
	localparam S_IDLE = 3'd0;
	localparam S_LOAD = 3'd1;
	localparam S_COMPUTE = 3'd2;
	localparam S_DRAIN = 3'd3;
	localparam S_DONE = 3'd4;
	localparam S_PRE_KICK = 3'd5; // Phát xung preload_start cho PSM
	localparam S_PRE_WAIT = 3'd6; // Chờ PSM dịch xong ac lần
	
	reg[2:0] state;
	
	assign state_dbg = state;
	
	always @(posedge i_clk or posedge i_reset) begin
		if(i_reset) begin
			state <= S_IDLE;
			
			o_feeder_start <= 1'b0;
			o_scan_en <= 1'b0;
			o_psm_clr <= 1'b0;
			o_psm_capture_en <= 1'b0;
			o_preload_start <= 1'b0;

			busy <= 1'b0;
			done <= 1'b0;
		end else begin
			o_feeder_start <= 1'b0;
			o_scan_en <= 1'b0;
			o_psm_clr <= 1'b0;
			o_psm_capture_en <= 1'b0;
			o_preload_start <= 1'b0;
			
			busy <= 1'b0;
			done <= 1'b0;
			
			case(state)
				S_IDLE: begin
					busy <= 1'b0;
					if(i_start) begin
						state <= S_LOAD;
						busy <= 1'b1;
					end
				end
				
				S_LOAD: begin
					busy <= 1'b1;
					// psm_clr: PSM chốt active_rows/cols của tile mới
					o_psm_clr <= 1'b1;
					state <= S_PRE_KICK;
				end

				S_PRE_KICK: begin
					busy <= 1'b1;
					o_preload_start <= 1'b1;
					state <= S_PRE_WAIT;
				end
				
				S_PRE_WAIT: begin
					busy <= 1'b1;
					// Shadow đã chứa giá trị khởi tạo thì mới cho feeder chạy
					if(i_preload_done)begin
						o_feeder_start <= 1'b1;
						state <= S_COMPUTE;
					end
				end

				S_COMPUTE: begin
					busy <= 1'b1;
					
					// Feeder + SA tự chạy bằng metadata
					if(i_sa_tile_done) begin
						state <= S_DRAIN;
					end
				end
				
				S_DRAIN: begin
					busy <= 1'b1;
					
					o_scan_en <= 1'b1;
					o_psm_capture_en <= 1'b1;
					
					if(i_psm_capture_done) begin
						state <= S_DONE;
					end
				end
				
				S_DONE: begin
					busy <= 1'b0;
					done <= 1'b1;
					
					if(!i_start) begin
						state <= S_IDLE;
					end
				end
				
				default: begin
					state <= S_IDLE;
				end
			endcase
		end
	end
endmodule
 
`default_nettype wire
