import sys
sys.path.append('..')
from hex.pytorch.graph_hex_board import GraphHexBoard

board = GraphHexBoard.new_vortex_board(11)
board.plot()