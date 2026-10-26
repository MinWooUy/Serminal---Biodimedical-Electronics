package gvu_config_pkg;
    parameter int DATA_WIDTH = 16;
    parameter int MAX_VECTOR_WIDTH = 128;
    parameter int VECTOR_LANES = 8;
    parameter int NUM_VECTOR_REGS = 32;
    parameter int NUM_SCRATCHPAD_BANKS = 1;
    parameter int SCRATCHPAD_DEPTH = 1024;
    parameter int REG_ADDR_WIDTH = $clog2((NUM_VECTOR_REGS < 2) ? 2 : NUM_VECTOR_REGS);
    parameter int MEM_ADDR_WIDTH = $clog2(((NUM_SCRATCHPAD_BANKS * SCRATCHPAD_DEPTH) < 2) ? 2 : (NUM_SCRATCHPAD_BANKS * SCRATCHPAD_DEPTH));
    parameter int OPCODE_WIDTH = 8;
    parameter int MAX_OUTSTANDING = 4;
    parameter int PIPELINE_STAGES = 1;
    parameter int ACC_WIDTH = 32;
    parameter int PSUM_WIDTH = 32;
    parameter int MAX_INFLIGHT_OPS = 8;
    parameter bit SUPPORT_INT8 = 1'b1;
    parameter bit SUPPORT_INT16 = 1'b1;
    parameter bit SUPPORT_INT32 = 1'b1;
    parameter bit SUPPORT_FP16 = 1'b0;
    parameter bit SUPPORT_FP32 = 1'b0;
    parameter bit SUPPORT_SATURATION = 1'b1;
    parameter bit SUPPORT_ROUNDING = 1'b1;

    localparam logic [OPCODE_WIDTH-1:0] OP_ADD = 'h00;
    localparam logic [OPCODE_WIDTH-1:0] OP_SUB = 'h01;
    localparam logic [OPCODE_WIDTH-1:0] OP_MUL = 'h02;
    localparam logic [OPCODE_WIDTH-1:0] OP_AND = 'h03;
    localparam logic [OPCODE_WIDTH-1:0] OP_OR  = 'h04;
    localparam logic [OPCODE_WIDTH-1:0] OP_XOR = 'h05;
    localparam logic [OPCODE_WIDTH-1:0] OP_SLT = 'h06;

    localparam logic [2:0] DATA_TYPE_INT8 = 3'd0;
    localparam logic [2:0] DATA_TYPE_INT16 = 3'd1;
    localparam logic [2:0] DATA_TYPE_INT32 = 3'd2;
    localparam logic [2:0] DATA_TYPE_FP16 = 3'd3;
    localparam logic [2:0] DATA_TYPE_FP32 = 3'd4;

    function automatic bit config_is_valid(
        input int data_width,
        input int max_vector_width,
        input int vector_lanes,
        input int num_vector_regs,
        input int num_scratchpad_banks,
        input int scratchpad_depth,
        input int max_outstanding,
        input int pipeline_stages,
        input int acc_width,
        input int psum_width,
        input int max_inflight_ops,
        input bit support_int8,
        input bit support_int16,
        input bit support_int32,
        input bit support_fp16,
        input bit support_fp32
    );
        config_is_valid = (data_width > 0) &&
                          (max_vector_width >= data_width) &&
                          (vector_lanes > 0) &&
                          (max_vector_width >= (data_width * vector_lanes)) &&
                          (num_vector_regs > 0) &&
                          (num_scratchpad_banks > 0) &&
                          (scratchpad_depth > 0) &&
                          (max_outstanding > 0) &&
                          (pipeline_stages > 0) &&
                          (acc_width >= data_width) &&
                          (psum_width > 0) &&
                          (max_inflight_ops > 0) &&
                          (support_int8 || support_int16 || support_int32 || support_fp16 || support_fp32);
    endfunction
endpackage
