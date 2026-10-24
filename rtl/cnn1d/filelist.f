// Danh sách nguồn RTL của accelerator Conv1D (đường dẫn tính từ rtl/cnn1d/)
// Dùng: vlog -f filelist.f tb/<tên_tb>.v

// Số học: bộ nhân Wallace/Baugh-Wooley có dấu, cộng CLA
arith/half_adder.v
arith/full_adder.v
arith/PartialFullAdder.v
arith/CarryLookaheadAdder_4_bit.v
arith/CarryLookaheadAdder_8_bit.v
arith/CarryLookaheadAdder_16_bit.v
arith/CarryLookaheadAdder_32_bit.v
arith/final_carry_lookahead_adder.v
arith/baugh_wooley_partial_product.v
arith/wallace_tree_reducer.v
arith/wallace_tree_8bit.v
arith/wallace_tree_multiplier.v

// Processing element
pe/WeightRegister.v
pe/ImageRegister.v
pe/Mul_Add_pipe.v
pe/Accumulator.v
pe/MAC.v
pe/shadow_register.v
pe/ProcessingElement.v
pe/scan_endpoint_mux.v
pe/pe_top.v

// Lõi tính toán: systolic array, PSM, FSM
core/pe_array.v
core/partial_sum_manager.v
core/cnn_control_fsm.v
core/cnn_compute_core.v

// Feeder ifmap/weight
feeder/FIFO_param.v
feeder/feed_delay.v
feeder/feed_lane.v
feeder/ifmap_feeder.v
feeder/weight_feeder.v

// Bộ nhớ tensor
mem/tensor_sram.v
mem/tensor_pingpong_buffer.v

// Conv1D: AGU, loader, tile top, layer controller
conv1d/conv1d_address_generating.v
conv1d/conv1d_ifmap_loader.v
conv1d/conv1d_weight_loader.v
conv1d/conv1d_ifmap_feeder_top.v
conv1d/conv1d_weight_feeder_top.v
conv1d/conv1d_tile_top.v
conv1d/conv1d_layer_controller.v

// Host UART 8N1, CRC and command transport
comm/crc/crc16_ccitt_next.v
comm/crc/crc_generator.v
comm/crc/crc_checker.v
comm/uart/uart_tx.v
comm/uart/uart_rx.v
comm/protocol/packet_rx.v
comm/protocol/packet_tx.v
comm/uart/uart_crc_top.v
comm/protocol/host_command_controller.v
top/ecg_accel_top.v

// Quantized convolution output and streaming pooling
post/conv1d_post_process.v
post/maxpool1d2.v
