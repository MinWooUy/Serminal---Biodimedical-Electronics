`timescale 1ns / 1ps
import gvu_config_pkg::*;

module gvu_config_check #(
    parameter int DATA_WIDTH = gvu_config_pkg::DATA_WIDTH,
    parameter int MAX_VECTOR_WIDTH = gvu_config_pkg::MAX_VECTOR_WIDTH,
    parameter int VECTOR_LANES = gvu_config_pkg::VECTOR_LANES,
    parameter int NUM_VECTOR_REGS = gvu_config_pkg::NUM_VECTOR_REGS,
    parameter int NUM_SCRATCHPAD_BANKS = gvu_config_pkg::NUM_SCRATCHPAD_BANKS,
    parameter int SCRATCHPAD_DEPTH = gvu_config_pkg::SCRATCHPAD_DEPTH,
    parameter int MAX_OUTSTANDING = gvu_config_pkg::MAX_OUTSTANDING,
    parameter int PIPELINE_STAGES = gvu_config_pkg::PIPELINE_STAGES,
    parameter int ACC_WIDTH = gvu_config_pkg::ACC_WIDTH,
    parameter int PSUM_WIDTH = gvu_config_pkg::PSUM_WIDTH,
    parameter int MAX_INFLIGHT_OPS = gvu_config_pkg::MAX_INFLIGHT_OPS,
    parameter bit SUPPORT_INT8 = gvu_config_pkg::SUPPORT_INT8,
    parameter bit SUPPORT_INT16 = gvu_config_pkg::SUPPORT_INT16,
    parameter bit SUPPORT_INT32 = gvu_config_pkg::SUPPORT_INT32,
    parameter bit SUPPORT_FP16 = gvu_config_pkg::SUPPORT_FP16,
    parameter bit SUPPORT_FP32 = gvu_config_pkg::SUPPORT_FP32
);
    initial begin
        if (!gvu_config_pkg::config_is_valid(
            DATA_WIDTH, MAX_VECTOR_WIDTH, VECTOR_LANES, NUM_VECTOR_REGS,
            NUM_SCRATCHPAD_BANKS, SCRATCHPAD_DEPTH, MAX_OUTSTANDING,
            PIPELINE_STAGES, ACC_WIDTH, PSUM_WIDTH, MAX_INFLIGHT_OPS,
            SUPPORT_INT8, SUPPORT_INT16, SUPPORT_INT32, SUPPORT_FP16, SUPPORT_FP32)) begin
            $fatal(1, "Invalid GVU configuration: check widths, lanes, memories, pipeline and enabled data types");
        end
    end
endmodule

module gvu_pipeline_controller #(
    parameter int OPCODE_WIDTH = gvu_config_pkg::OPCODE_WIDTH,
    parameter int VECTOR_LEN = 1
) (
    input  logic clk,
    input  logic rst_n,
    input  logic start,
    input  logic [OPCODE_WIDTH-1:0] opcode,
    output logic busy,
    output logic issue_valid,
    output logic [$clog2((VECTOR_LEN < 2) ? 2 : VECTOR_LEN)-1:0] lane_index,
    output logic [OPCODE_WIDTH-1:0] issue_opcode,
    input  logic issue_ready,
    input logic done,
    input logic dependency_stall,
    input logic resource_stall,
    input logic flush,
    output logic stall,
    output logic complete,
    output logic error
);
    localparam int INDEX_WIDTH = $clog2((VECTOR_LEN < 2) ? 2 : VECTOR_LEN);

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            busy <= 1'b0;
            issue_valid <= 1'b0;
            lane_index <= '0;
            issue_opcode <= '0;
            stall <= 1'b0;
            complete <= 1'b0;
            error <= 1'b0;
        end else begin
            issue_valid <= 1'b0;
            complete <= 1'b0;
            error <= 1'b0;
            stall <= dependency_stall || resource_stall;
            if (flush) begin
                busy <= 1'b0;
                lane_index <= '0;
            end else if (start && !busy && !dependency_stall && !resource_stall) begin
                busy <= 1'b1;
                lane_index <= '0;
                issue_opcode <= opcode;
                issue_valid <= 1'b1;
            end else if (busy && issue_ready && !dependency_stall && !resource_stall) begin
                if (lane_index == VECTOR_LEN - 1) begin
                    busy <= 1'b0;
                    complete <= 1'b1;
                end else begin
                    lane_index <= lane_index + 1'b1;
                    issue_valid <= 1'b1;
                end
            end
            if (done) begin
                busy <= 1'b0;
                complete <= 1'b1;
            end
        end
    end
endmodule

module gvu_instruction_decode_if #(
    parameter int OPCODE_WIDTH = gvu_config_pkg::OPCODE_WIDTH,
    parameter int REGISTER_INDEX_WIDTH = gvu_config_pkg::REG_ADDR_WIDTH,
    parameter int IMMEDIATE_WIDTH = 12,
    parameter int INSTRUCTION_WIDTH = OPCODE_WIDTH + (3 * REGISTER_INDEX_WIDTH) + IMMEDIATE_WIDTH
) (
    input  logic [INSTRUCTION_WIDTH-1:0] instruction,
    input  logic valid_in,
    output logic [OPCODE_WIDTH-1:0] opcode,
    output logic [REGISTER_INDEX_WIDTH-1:0] rd,
    output logic [REGISTER_INDEX_WIDTH-1:0] rs1,
    output logic [REGISTER_INDEX_WIDTH-1:0] rs2,
    output logic [IMMEDIATE_WIDTH-1:0] immediate,
    output logic valid_out,
    output logic illegal_instruction
);
    always_comb begin
        opcode = instruction[OPCODE_WIDTH-1:0];
        rd = instruction[OPCODE_WIDTH +: REGISTER_INDEX_WIDTH];
        rs1 = instruction[OPCODE_WIDTH + REGISTER_INDEX_WIDTH +: REGISTER_INDEX_WIDTH];
        rs2 = instruction[OPCODE_WIDTH + 2 * REGISTER_INDEX_WIDTH +: REGISTER_INDEX_WIDTH];
        immediate = instruction[OPCODE_WIDTH + (3 * REGISTER_INDEX_WIDTH) +: IMMEDIATE_WIDTH];
        valid_out = valid_in;
        illegal_instruction = valid_in && (opcode > gvu_config_pkg::OP_SLT);
    end
endmodule

module gvu_main_controller_if #(
    parameter int CONTROL_WIDTH = 8
) (
    input  logic clk,
    input  logic rst_n,
    input  logic cmd_valid,
    input  logic [CONTROL_WIDTH-1:0] cmd,
    output logic cmd_ready,
    output logic [CONTROL_WIDTH-1:0] control_state
);
    assign cmd_ready = 1'b1;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) control_state <= '0;
        else if (cmd_valid) control_state <= cmd;
    end
endmodule

module gvu_stream_switch #(
    parameter int LANES = 4,
    parameter int DATA_WIDTH = 16,
    parameter int SELECT_WIDTH = $clog2((LANES < 2) ? 2 : LANES)
) (
    input  logic [LANES-1:0][DATA_WIDTH-1:0] in_data,
    input  logic [LANES-1:0] in_valid,
    output logic [LANES-1:0] in_ready,
    input  logic [SELECT_WIDTH-1:0] select,
    output logic [DATA_WIDTH-1:0] out_data,
    output logic out_valid,
    input  logic out_ready
);
    always_comb begin
        out_data = '0;
        out_valid = 1'b0;
        in_ready = '0;
        if (select < LANES) begin
            out_data = in_data[select];
            out_valid = in_valid[select];
            in_ready[select] = out_ready;
        end
    end
endmodule

module gvu_vector_register_file #(
    parameter int LANES = gvu_config_pkg::VECTOR_LANES,
    parameter int DATA_WIDTH = gvu_config_pkg::DATA_WIDTH,
    parameter int REGISTER_COUNT = gvu_config_pkg::NUM_VECTOR_REGS,
    parameter int REGISTER_INDEX_WIDTH = $clog2((REGISTER_COUNT < 2) ? 2 : REGISTER_COUNT)
) (
    input  logic clk,
    input  logic rst_n,
    input  logic write_en,
    input  logic [REGISTER_INDEX_WIDTH-1:0] write_index,
    input  logic [LANES-1:0][DATA_WIDTH-1:0] write_data,
    input  logic [LANES-1:0] write_lane_mask,
    input  logic [REGISTER_INDEX_WIDTH-1:0] read_index_a,
    input  logic [REGISTER_INDEX_WIDTH-1:0] read_index_b,
    output logic [LANES-1:0][DATA_WIDTH-1:0] read_data_a,
    output logic [LANES-1:0][DATA_WIDTH-1:0] read_data_b
);
    logic [LANES-1:0][DATA_WIDTH-1:0] registers [0:REGISTER_COUNT-1];
    integer index;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (index = 0; index < REGISTER_COUNT; index = index + 1) registers[index] <= '0;
        end else if (write_en && (write_index < REGISTER_COUNT)) begin
            for (index = 0; index < LANES; index = index + 1) begin
                if (write_lane_mask[index]) registers[write_index][index] <= write_data[index];
            end
        end
    end
    always_comb begin
        read_data_a = (read_index_a < REGISTER_COUNT) ? registers[read_index_a] : '0;
        read_data_b = (read_index_b < REGISTER_COUNT) ? registers[read_index_b] : '0;
    end
endmodule

module gvu_scratchpad_memory #(
    parameter int DATA_WIDTH = gvu_config_pkg::DATA_WIDTH,
    parameter int NUM_BANKS = gvu_config_pkg::NUM_SCRATCHPAD_BANKS,
    parameter int DEPTH = gvu_config_pkg::SCRATCHPAD_DEPTH,
    parameter int ADDRESS_WIDTH = $clog2((NUM_BANKS * DEPTH < 2) ? 2 : NUM_BANKS * DEPTH)
) (
    input  logic clk,
    input  logic write_en,
    input  logic [ADDRESS_WIDTH-1:0] write_address,
    input  logic [DATA_WIDTH-1:0] write_data,
    input  logic read_en,
    input  logic [ADDRESS_WIDTH-1:0] read_address,
    output logic [DATA_WIDTH-1:0] read_data,
    output logic read_valid
);
    localparam int BANK_WIDTH = $clog2((NUM_BANKS < 2) ? 2 : NUM_BANKS);
    logic [DATA_WIDTH-1:0] memory [0:NUM_BANKS-1][0:DEPTH-1];
    integer bank_index;
    integer element_index;
    always_ff @(posedge clk) begin
        bank_index = write_address / DEPTH;
        element_index = write_address % DEPTH;
        if (write_en && (bank_index < NUM_BANKS)) memory[bank_index][element_index] <= write_data;
        read_valid <= read_en;
        bank_index = read_address / DEPTH;
        element_index = read_address % DEPTH;
        if (read_en && (bank_index < NUM_BANKS)) read_data <= memory[bank_index][element_index];
        else read_data <= '0;
    end
endmodule

module gvu_vlsu #(
    parameter int DATA_WIDTH = gvu_config_pkg::DATA_WIDTH,
    parameter int ADDRESS_WIDTH = gvu_config_pkg::MEM_ADDR_WIDTH,
    parameter int MAX_OUTSTANDING = gvu_config_pkg::MAX_OUTSTANDING
) (
    input  logic clk,
    input  logic rst_n,
    input  logic load_valid,
    output logic load_ready,
    input  logic [ADDRESS_WIDTH-1:0] load_address,
    input  logic [DATA_WIDTH-1:0] load_data,
    output logic mem_write_en,
    output logic [ADDRESS_WIDTH-1:0] mem_write_address,
    output logic [DATA_WIDTH-1:0] mem_write_data,
    input  logic store_valid,
    output logic store_ready,
    input  logic [ADDRESS_WIDTH-1:0] store_address,
    input  logic [DATA_WIDTH-1:0] store_data,
    output logic mem_read_en,
    output logic [ADDRESS_WIDTH-1:0] mem_read_address
);
    assign load_ready = 1'b1;
    assign store_ready = 1'b1;
    assign mem_write_en = load_valid;
    assign mem_write_address = load_address;
    assign mem_write_data = load_data;
    assign mem_read_en = store_valid;
    assign mem_read_address = store_address;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            // Handshake is combinational; reset is reserved for future outstanding requests.
        end
    end
endmodule

module gvu_functional_unit #(
    parameter int LANES = gvu_config_pkg::VECTOR_LANES,
    parameter int DATA_WIDTH = gvu_config_pkg::DATA_WIDTH,
    parameter int ACC_WIDTH = gvu_config_pkg::ACC_WIDTH,
    parameter int OPCODE_WIDTH = gvu_config_pkg::OPCODE_WIDTH,
    parameter int FU_COUNT = 1,
    parameter int PIPELINE_STAGES = gvu_config_pkg::PIPELINE_STAGES,
    parameter int MAX_INFLIGHT_OPS = gvu_config_pkg::MAX_INFLIGHT_OPS,
    parameter bit SUPPORT_SATURATION = gvu_config_pkg::SUPPORT_SATURATION,
    parameter bit SUPPORT_ROUNDING = gvu_config_pkg::SUPPORT_ROUNDING
) (
    input  logic clk,
    input  logic rst_n,
    input  logic valid_in,
    output logic ready_in,
    input  logic [OPCODE_WIDTH-1:0] opcode,
    input  logic [LANES-1:0][DATA_WIDTH-1:0] operand_a,
    input  logic [LANES-1:0][DATA_WIDTH-1:0] operand_b,
    output logic valid_out,
    input  logic ready_out,
    output logic [LANES-1:0][ACC_WIDTH-1:0] result
);
    logic [LANES-1:0][ACC_WIDTH-1:0] next_result;
    integer lane;
    assign ready_in = ready_out || !valid_out;
    always_comb begin
        next_result = '0;
        for (lane = 0; lane < LANES; lane = lane + 1) begin
            case (opcode)
                gvu_config_pkg::OP_ADD: next_result[lane] = $signed(operand_a[lane]) + $signed(operand_b[lane]);
                gvu_config_pkg::OP_SUB: next_result[lane] = $signed(operand_a[lane]) - $signed(operand_b[lane]);
                gvu_config_pkg::OP_MUL: next_result[lane] = $signed(operand_a[lane]) * $signed(operand_b[lane]);
                gvu_config_pkg::OP_AND: next_result[lane] = operand_a[lane] & operand_b[lane];
                gvu_config_pkg::OP_OR:  next_result[lane] = operand_a[lane] | operand_b[lane];
                gvu_config_pkg::OP_XOR: next_result[lane] = operand_a[lane] ^ operand_b[lane];
                gvu_config_pkg::OP_SLT: next_result[lane] = ($signed(operand_a[lane]) < $signed(operand_b[lane])) ? 1 : 0;
                default: next_result[lane] = operand_a[lane];
            endcase
        end
    end
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_out <= 1'b0;
            result <= '0;
        end else if (ready_in) begin
            valid_out <= valid_in;
            if (valid_in) result <= next_result;
        end
    end
endmodule

module gvu_dma_if #(
    parameter int DATA_WIDTH = gvu_config_pkg::DATA_WIDTH,
    parameter int ADDRESS_WIDTH = gvu_config_pkg::MEM_ADDR_WIDTH,
    parameter int MAX_OUTSTANDING = gvu_config_pkg::MAX_OUTSTANDING
) (
    input  logic clk,
    input  logic rst_n,
    input  logic request_valid,
    output logic request_ready,
    input  logic request_write,
    input  logic [ADDRESS_WIDTH-1:0] request_address,
    input  logic [DATA_WIDTH-1:0] request_wdata,
    output logic response_valid,
    input  logic response_ready,
    output logic [DATA_WIDTH-1:0] response_rdata,
    output logic dma_valid,
    input  logic dma_ready,
    output logic dma_write,
    output logic [ADDRESS_WIDTH-1:0] dma_address,
    output logic [DATA_WIDTH-1:0] dma_wdata,
    input  logic [DATA_WIDTH-1:0] dma_rdata
);
    assign request_ready = dma_ready && !response_valid;
    assign dma_valid = request_valid && request_ready;
    assign dma_write = request_write;
    assign dma_address = request_address;
    assign dma_wdata = request_wdata;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            response_valid <= 1'b0;
            response_rdata <= '0;
        end else begin
            if (dma_valid && !request_write) begin
                response_valid <= 1'b1;
                response_rdata <= dma_rdata;
            end else if (response_ready) response_valid <= 1'b0;
        end
    end
endmodule

module gvu_sram_c_if #(
    parameter int DATA_WIDTH = gvu_config_pkg::ACC_WIDTH,
    parameter int ADDRESS_WIDTH = gvu_config_pkg::MEM_ADDR_WIDTH
) (
    input  logic clk,
    input  logic rst_n,
    input  logic valid_in,
    input  logic [ADDRESS_WIDTH-1:0] address_in,
    input  logic [DATA_WIDTH-1:0] data_in,
    output logic ready_in,
    output logic sram_we,
    output logic [ADDRESS_WIDTH-1:0] sram_address,
    output logic [DATA_WIDTH-1:0] sram_wdata
);
    assign ready_in = 1'b1;
    assign sram_we = valid_in;
    assign sram_address = address_in;
    assign sram_wdata = data_in;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            // External SRAM owns the storage; this interface only registers no state.
        end
    end
endmodule

module gvu_psum_if #(
    parameter int LANES = gvu_config_pkg::VECTOR_LANES,
    parameter int DATA_WIDTH = gvu_config_pkg::PSUM_WIDTH,
    parameter int ACC_WIDTH = gvu_config_pkg::ACC_WIDTH
) (
    input  logic [LANES-1:0][DATA_WIDTH-1:0] partial_sum_in,
    input  logic [LANES-1:0][ACC_WIDTH-1:0] accumulation,
    input  logic accumulate_en,
    output logic [LANES-1:0][ACC_WIDTH-1:0] partial_sum_out
);
    integer lane;
    always_comb begin
        for (lane = 0; lane < LANES; lane = lane + 1) begin
            if (accumulate_en) partial_sum_out[lane] = $signed(partial_sum_in[lane]) + $signed(accumulation[lane]);
            else partial_sum_out[lane] = partial_sum_in[lane];
        end
    end
endmodule

module gvu #(
    parameter int DATA_WIDTH = gvu_config_pkg::DATA_WIDTH,
    parameter int MAX_VECTOR_WIDTH = gvu_config_pkg::MAX_VECTOR_WIDTH,
    parameter int LANES = gvu_config_pkg::VECTOR_LANES,
    parameter int REGISTER_COUNT = gvu_config_pkg::NUM_VECTOR_REGS,
    parameter int NUM_SCRATCHPAD_BANKS = gvu_config_pkg::NUM_SCRATCHPAD_BANKS,
    parameter int SCRATCHPAD_DEPTH = gvu_config_pkg::SCRATCHPAD_DEPTH,
    parameter int ACC_WIDTH = gvu_config_pkg::ACC_WIDTH,
    parameter int PSUM_WIDTH = gvu_config_pkg::PSUM_WIDTH,
    parameter int MAX_OUTSTANDING = gvu_config_pkg::MAX_OUTSTANDING,
    parameter int PIPELINE_STAGES = gvu_config_pkg::PIPELINE_STAGES,
    parameter int MAX_INFLIGHT_OPS = gvu_config_pkg::MAX_INFLIGHT_OPS,
    parameter bit SUPPORT_INT8 = gvu_config_pkg::SUPPORT_INT8,
    parameter bit SUPPORT_INT16 = gvu_config_pkg::SUPPORT_INT16,
    parameter bit SUPPORT_INT32 = gvu_config_pkg::SUPPORT_INT32,
    parameter bit SUPPORT_FP16 = gvu_config_pkg::SUPPORT_FP16,
    parameter bit SUPPORT_FP32 = gvu_config_pkg::SUPPORT_FP32,
    parameter bit SUPPORT_SATURATION = gvu_config_pkg::SUPPORT_SATURATION,
    parameter bit SUPPORT_ROUNDING = gvu_config_pkg::SUPPORT_ROUNDING,
    parameter int FU_COUNT = 1,
    parameter int OPCODE_WIDTH = gvu_config_pkg::OPCODE_WIDTH,
    parameter int ADDRESS_WIDTH = gvu_config_pkg::MEM_ADDR_WIDTH,
    parameter int REGISTER_INDEX_WIDTH = $clog2((REGISTER_COUNT < 2) ? 2 : REGISTER_COUNT),
    parameter int REG_ADDR_WIDTH = REGISTER_INDEX_WIDTH,
    parameter int MEM_ADDR_WIDTH = ADDRESS_WIDTH
) (
    input  logic clk,
    input  logic rst_n,
    input  logic instruction_valid,
    input  logic [OPCODE_WIDTH-1:0] instruction_opcode,
    input  logic [REGISTER_INDEX_WIDTH-1:0] instruction_rd,
    input  logic [REGISTER_INDEX_WIDTH-1:0] instruction_rs1,
    input  logic [REGISTER_INDEX_WIDTH-1:0] instruction_rs2,
    input  logic [LANES-1:0] instruction_lane_mask,
    input  logic [LANES-1:0][DATA_WIDTH-1:0] load_data,
    input  logic load_valid,
    output logic load_ready,
    output logic [LANES-1:0][ACC_WIDTH-1:0] result,
    output logic result_valid,
    output logic sram_c_we,
    output logic [ADDRESS_WIDTH-1:0] sram_c_address,
    output logic [LANES-1:0][ACC_WIDTH-1:0] sram_c_data
);
    logic [LANES-1:0][DATA_WIDTH-1:0] operands_a, operands_b;
    logic [LANES-1:0][DATA_WIDTH-1:0] rf_write_data;
    logic [LANES-1:0][ACC_WIDTH-1:0] fu_result;
    logic fu_ready;
    logic rf_write_en;

    gvu_config_check #(
        .DATA_WIDTH(DATA_WIDTH), .MAX_VECTOR_WIDTH(MAX_VECTOR_WIDTH), .VECTOR_LANES(LANES),
        .NUM_VECTOR_REGS(REGISTER_COUNT), .NUM_SCRATCHPAD_BANKS(NUM_SCRATCHPAD_BANKS),
        .SCRATCHPAD_DEPTH(SCRATCHPAD_DEPTH), .MAX_OUTSTANDING(MAX_OUTSTANDING),
        .PIPELINE_STAGES(PIPELINE_STAGES), .ACC_WIDTH(ACC_WIDTH), .PSUM_WIDTH(PSUM_WIDTH),
        .MAX_INFLIGHT_OPS(MAX_INFLIGHT_OPS), .SUPPORT_INT8(SUPPORT_INT8),
        .SUPPORT_INT16(SUPPORT_INT16), .SUPPORT_INT32(SUPPORT_INT32),
        .SUPPORT_FP16(SUPPORT_FP16), .SUPPORT_FP32(SUPPORT_FP32)
    ) config_check ();

    genvar lane;
    generate
        for (lane = 0; lane < LANES; lane = lane + 1) begin : gen_rf_write_data
            assign rf_write_data[lane] = fu_result[lane][DATA_WIDTH-1:0];
        end
    endgenerate

    gvu_vector_register_file #(.LANES(LANES), .DATA_WIDTH(DATA_WIDTH), .REGISTER_COUNT(REGISTER_COUNT), .REGISTER_INDEX_WIDTH(REG_ADDR_WIDTH)) rf (
        .clk(clk), .rst_n(rst_n), .write_en(rf_write_en), .write_index(instruction_rd), .write_data(rf_write_data), .write_lane_mask(instruction_lane_mask),
        .read_index_a(instruction_rs1), .read_index_b(instruction_rs2), .read_data_a(operands_a), .read_data_b(operands_b));
    gvu_functional_unit #(.LANES(LANES), .DATA_WIDTH(DATA_WIDTH), .ACC_WIDTH(ACC_WIDTH), .OPCODE_WIDTH(OPCODE_WIDTH), .FU_COUNT(FU_COUNT), .PIPELINE_STAGES(PIPELINE_STAGES), .MAX_INFLIGHT_OPS(MAX_INFLIGHT_OPS), .SUPPORT_SATURATION(SUPPORT_SATURATION), .SUPPORT_ROUNDING(SUPPORT_ROUNDING)) fu (
        .clk(clk), .rst_n(rst_n), .valid_in(instruction_valid), .ready_in(fu_ready), .opcode(instruction_opcode),
        .operand_a(operands_a), .operand_b(operands_b), .valid_out(result_valid), .ready_out(1'b1), .result(fu_result));
    assign load_ready = 1'b1;
    assign rf_write_en = result_valid;
    assign result = fu_result;
    assign sram_c_we = result_valid;
    assign sram_c_address = '0;
    assign sram_c_data = fu_result;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            // Top-level completion is represented by result_valid.
        end
    end
endmodule