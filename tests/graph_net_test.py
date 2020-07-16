import unittest
import os,sys; sys.path.insert(0, os.path.abspath('.'))
import torch

from hex.pytorch.board_graph import Board, PositionalEncoder
from hex.pytorch.graph_net import GraphNet
from utils import dotdict

class Test_GraphNet(unittest.TestCase):

    def test_model(self):
        b = Board(torch.zeros(4,4).long())
        b.np_pieces[1,0] = -1
        b.np_pieces[1,1] = -1
        b.np_pieces[2,1] = -1
        b.np_pieces[2,2] = -1
        b.np_pieces[0,2] = 1
        b.np_pieces[0,3] = 1
        b.np_pieces[3,2] = 1
        b.np_pieces[2,0] = 1

        args = dotdict({
            'lr': 0.001,
            'dropout': 0.3,
            'epochs': 10,
            'batch_size': 64,
            'cuda': torch.cuda.is_available(),
            'num_channels': 32,
            'res_blocks' : 5,
            'board_size' : 4,
            'expand_base' : 2,
            'attn_heads' : 1,
            'pos_encoding_sz' : 28,
            'readout_attn_heads' : 4
        })

        model = GraphNet(args)
        optimizer = torch.optim.Adam(model.parameters())
        p, v = model(b.np_pieces.unsqueeze(dim=0))
        

    def test_position_encoding(self):
        penc = PositionalEncoder(d_model=20, max_seq_len=100)
        x = torch.arange(50)
        x = penc(x)
