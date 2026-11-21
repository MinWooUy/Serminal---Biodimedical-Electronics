"""Byte-level software model of the layer_cmd_controller FPGA (no hardware needed).

Written from rtl/cnn1d/top/layer_cmd_controller.v + conv1d_layer_top.v (RTL ca7562e); it is NOT the RTL and
is only as good as that reading.  Use it to run the PC-side code (LayerHost, board_classifier, monitors)
without a board.  Real hardware results are the only proof for the bitstream.

    from python.link.layersim import LayerFpgaSim, LayerLink, HwConfig
    link = LayerLink(LayerFpgaSim(HwConfig(rows=3, cols=3, baud=921600)))   # transport: write/read/clock
"""
from .fpga_sim import LayerFpgaSim, LayerLink
from .sched import HwConfig

__all__ = ["LayerFpgaSim", "LayerLink", "HwConfig"]
