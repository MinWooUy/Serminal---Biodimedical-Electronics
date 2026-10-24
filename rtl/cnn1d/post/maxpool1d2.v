// MaxPool(kernel=2,stride=2). Supply consecutive positions for the SAME
// channel group. Bubbles allowed; clear between sequences/channel groups.
// An odd final position is discarded by clear, matching floor-length pooling.
module maxpool1d2 #(parameter CHANNELS=3)(
    input wire i_clk, i_reset, i_clear, i_valid,
    input wire [CHANNELS*8-1:0] i_data,
    output reg [CHANNELS*8-1:0] o_data,
    output reg o_valid
);
    reg pending;
    reg [CHANNELS*8-1:0] first;
    integer c;
    always @(posedge i_clk or posedge i_reset) begin
        if(i_reset) begin pending<=0; first<=0; o_data<=0; o_valid<=0; end
        else begin
            o_valid<=0;
            if(i_clear) pending<=0;
            else if(i_valid) begin
                pending<=!pending;
                if(!pending) first<=i_data;
                else begin
                    o_valid<=1;
                    for(c=0;c<CHANNELS;c=c+1)
                        o_data[c*8 +: 8] <= ($signed(first[c*8 +: 8]) > $signed(i_data[c*8 +: 8]))
                            ? first[c*8 +: 8] : i_data[c*8 +: 8];
                end
            end
        end
    end
endmodule
