import unittest
import os,sys; sys.path.insert(0, os.path.abspath('.'))
import torch

from hex.pytorch.board_graph import Board, PositionalEncoder
from hex.pytorch.graph_net import GraphNet
from utils import dotdict


def loss_pi(targets, outputs):
    return -torch.sum(targets * outputs) / targets.size()[0]

def loss_v(targets, outputs):
    return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]

class Test_GraphNet(unittest.TestCase):

    def test_model(self):
        args = dotdict({
            'lr': 0.001,
            'dropout': 0.3,
            'epochs': 10,
            'batch_size': 1,
            'cuda': torch.cuda.is_available(),
            'num_channels': 32,
            'res_blocks' : 5,
            'board_size' : 4,
            'expand_base' : 2,
            'attn_heads' : 1,
            'id_embedding_sz' : 28,
            'readout_attn_heads' : 4
        })
        x = torch.zeros(args.batch_size, args.board_size, args.board_size).long()
        x[0,1,0] = -1
        x[0,1,1] = -1
        x[0,2,1] = -1
        x[0,2,2] = -1
        x[0,0,2] = 1
        x[0,0,3] = 1
        x[0,3,2] = 1
        x[0,2,0] = 1
        y_p = torch.rand(args.batch_size, args.board_size**2).softmax(dim=1)
        y_v = torch.rand(args.batch_size, 1).tanh()

        model = GraphNet(args)
        optimizer = torch.optim.Adam(model.parameters())
        p, v = model(x)
        loss = loss_pi(y_p, p) + loss_v(y_v, v)
        loss.backward()
        optimizer.step()
        
        self.assertEqual(p.size(0), args.batch_size)
        self.assertEqual(p.size(1), args.board_size**2)
        self.assertEqual(v.size(0), args.batch_size)
        self.assertEqual(v.size(1), 1)
          
    def test_model_batch(self):
        """ test model with a batch of boards
        """

        args = dotdict({
            'lr': 0.001,
            'dropout': 0.3,
            'epochs': 10,
            'batch_size': 16,
            'cuda': torch.cuda.is_available(),
            'num_channels': 32,
            'res_blocks' : 5,
            'board_size' : 7,
            'expand_base' : 2,
            'attn_heads' : 1,
            'id_embedding_sz' : 28,
            'readout_attn_heads' : 4
        })

        x = torch.zeros(args.batch_size, args.board_size, args.board_size).long()
        x[0,1,0] = -1
        x[0,1,1] = -1
        x[0,2,1] = -1
        x[0,2,2] = -1
        x[0,0,2] = 1
        x[0,0,3] = 1
        x[0,3,2] = 1
        x[0,2,0] = 1
        x[1,4,0] = -1
        x[1,5,1] = -1
        x[1,6,1] = -1
        x[1,2,2] = -1
        x[1,3,2] = 1
        x[1,0,3] = 1
        x[1,3,2] = 1
        x[1,2,0] = 1

        y_p = torch.rand(args.batch_size, args.board_size**2).softmax(dim=1)
        y_v = torch.rand(args.batch_size, 1).tanh()

        model = GraphNet(args)
        optimizer = torch.optim.Adam(model.parameters())
        p, v = model(x)
        loss = loss_pi(y_p, p) + loss_v(y_v, v)
        loss.backward()
        optimizer.step()

        self.assertEqual(p.size(0), args.batch_size)
        self.assertEqual(p.size(1), args.board_size**2)
        self.assertEqual(v.size(0), args.batch_size)
        self.assertEqual(v.size(1), 1)

    def test_position_encoding(self):
        penc = PositionalEncoder(d_model=20, max_seq_len=100)
        x = torch.arange(50)
        x = penc(x)
