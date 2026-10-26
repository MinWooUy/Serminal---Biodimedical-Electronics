module packet_tx #(parameter MAX_PAYLOAD=256)(
    input wire clk, reset, start,
    input wire [7:0] cmd, seq,
    input wire [15:0] length,
    input wire [MAX_PAYLOAD*8-1:0] payload,
    output reg busy,
    output wire byte_valid,
    output reg [7:0] byte_data,
    input wire byte_ready
);
    reg [7:0] cmd_q,seq_q;
    reg [15:0] len_q, pos, crc;
    reg [MAX_PAYLOAD*8-1:0] body;
    wire [15:0] next_crc;
    crc16_ccitt_next u_next(crc,byte_data,next_crc);
    assign byte_valid=busy;
    always @* begin
        case(pos)
            0: byte_data=8'ha5;
            1: byte_data=8'h5a;
            2: byte_data=cmd_q;
            3: byte_data=seq_q;
            4: byte_data=len_q[7:0];
            5: byte_data=len_q[15:8];
            default: begin
                if(pos<6+len_q) byte_data=body[(pos-6)*8 +: 8];
                else if(pos==6+len_q) byte_data=crc[7:0];
                else byte_data=crc[15:8];
            end
        endcase
    end
    always @(posedge clk or posedge reset) begin
        if(reset) begin busy<=0; pos<=0; crc<=16'hffff; body<=0;
            cmd_q<=0;seq_q<=0;len_q<=0; end
        else if(!busy) begin
            if(start && length<=MAX_PAYLOAD) begin
                busy<=1;pos<=0;crc<=16'hffff;
                body<=payload;cmd_q<=cmd;seq_q<=seq;len_q<=length;
            end
        end else if(byte_ready) begin
            if(pos>=2 && pos<6+len_q) crc<=next_crc;
            if(pos==7+len_q) busy<=0;
            else pos<=pos+1'b1;
        end
    end
endmodule
