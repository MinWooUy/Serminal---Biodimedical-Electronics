// Frame: A5 5A CMD SEQ LEN_LO LEN_HI PAYLOAD CRC_LO CRC_HI.
// CRC covers CMD through PAYLOAD. Payload is published only after CRC passes.
module packet_rx #(
    parameter MAX_PAYLOAD=256,
    parameter TIMEOUT_CLKS=5000000
)(
    input wire clk, reset, enable,
    input wire [7:0] byte_data,
    input wire byte_valid, frame_error,
    output reg [7:0] cmd, seq,
    output reg [15:0] length,
    output reg [MAX_PAYLOAD*8-1:0] payload,
    output reg packet_valid, packet_error
);
    localparam SOF0=0, SOF1=1, CMD=2, SEQ=3, LEN0=4, LEN1=5,
               BODY=6, CRC0=7, CRC1=8;
    reg [3:0] state;
    reg [15:0] crc, index;
    reg [7:0] crc_low;
    localparam TW=(TIMEOUT_CLKS<=1)?1:$clog2(TIMEOUT_CLKS);
    reg [TW-1:0] timer;
    wire [15:0] next_crc;
    crc16_ccitt_next u_next(crc, byte_data, next_crc);
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            state<=SOF0; crc<=16'hffff; index<=0; timer<=0;
            cmd<=0; seq<=0; length<=0; payload<=0; crc_low<=0;
            packet_valid<=0; packet_error<=0;
        end else begin
            packet_valid<=0; packet_error<=0;
            if (!enable || frame_error) begin state<=SOF0; timer<=0; end
            else if (byte_valid) begin
                timer<=0;
                case(state)
                    SOF0: if(byte_data==8'ha5) state<=SOF1;
                    SOF1: if(byte_data==8'h5a) begin
                        state<=CMD; crc<=16'hffff; cmd<=0; seq<=0;
                    end else if(byte_data!=8'ha5) state<=SOF0;
                    CMD: begin cmd<=byte_data; crc<=next_crc; state<=SEQ; end
                    SEQ: begin seq<=byte_data; crc<=next_crc; state<=LEN0; end
                    LEN0: begin length[7:0]<=byte_data; crc<=next_crc; state<=LEN1; end
                    LEN1: begin
                        length[15:8]<=byte_data; crc<=next_crc; index<=0;
                        if ({byte_data,length[7:0]} > MAX_PAYLOAD) begin
                            state<=SOF0; packet_error<=1;
                        end else if ({byte_data,length[7:0]}==0) state<=CRC0;
                        else state<=BODY;
                    end
                    BODY: begin
                        payload[index*8 +: 8]<=byte_data; crc<=next_crc;
                        if(index==length-1) state<=CRC0;
                        else index<=index+1'b1;
                    end
                    CRC0: begin crc_low<=byte_data; state<=CRC1; end
                    CRC1: begin
                        state<=SOF0;
                        if ({byte_data,crc_low}==crc) packet_valid<=1;
                        else packet_error<=1;
                    end
                    default: state<=SOF0;
                endcase
            end else if(state!=SOF0) begin
                if(timer==TIMEOUT_CLKS-1) begin state<=SOF0; timer<=0; end
                else timer<=timer+1'b1;
            end else timer<=0;
        end
    end
endmodule
