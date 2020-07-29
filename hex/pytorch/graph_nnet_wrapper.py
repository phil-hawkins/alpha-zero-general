import os
import sys
import time
import math

import numpy as np
from tqdm import tqdm

sys.path.append('../')
from utils import *
from NeuralNet import NeuralNet

import torch
import torch.optim as optim

#from .Connect4NNet import Connect4NNet as c4nnet
from .graph_net import NativeGraphNet as GraphNet
from .board_graph import Board, BoardGraph, PlayerGraph, IdentifierEncoder, ZeroIdentifierEncoder, RandomIdentifierEncoder

args = dotdict({
    'dropout': 0.3,
    'epochs': 10,
    'batch_size': 64,
    'cuda': torch.cuda.is_available(),
    'num_channels': 128,#32,
    'res_blocks' : 5,
    'in_channels' : 3                   # 0/1/2 - black/white/empty
})

class FakeNNet(NeuralNet):
    """ fake neural network that does random predictions for pitting an oponent against pure MCTS
    """
    def __init__(self, game, net_type=None, value_function=None):
        self.game = game
        self.value_function = value_function if value_function else lambda x : 0.

    def predict(self, board):
        valids = self.game.getValidMoves(board)
        pi = np.zeros_like(valids, dtype=np.float32)
        valids_ndx = np.nonzero(valids)[0]
        np.random.shuffle(valids_ndx)
        action_ndx = valids_ndx[0]
        pi[action_ndx] = 1.0
        v = self.value_function(board)

        return pi, v    

def value_from_shortest_path(board):
    """ takes a matrix representation of the board and calculates a state value based on a 
    comparison of shortest paths of each player
    """
    board_size = board.shape[0]
    bg = BoardGraph.from_graph_board(board)
    g_p1, g_p2 = PlayerGraph.from_board_graph(bg, 1), PlayerGraph.from_board_graph(bg, -1)
    sp_p1, sp_p2 = g_p1.shortest_path(), g_p2.shortest_path()

    if sp_p1 == 0:
        v = 1.0
    elif sp_p2 == 0:
        v = -1.0
    else:
        v = (sp_p2 - sp_p1) / board_size

    return v

class GraphNNetWrapper(NeuralNet):
    def __init__(self, game, net_type="base_gat"):
        def random_id_args(d_model):
            args['num_channels'] = 32
            args['expand_base'] = 2
            args['attn_heads'] = 1
            args['readout_attn_heads'] = 4
            args['id_encoder'] = RandomIdentifierEncoder(d_model=d_model)

        self.net_type = net_type
        if self.net_type == "base_gat":        
            args['num_channels'] = 32
            args['expand_base'] = 2
            args['attn_heads'] = 1
            args['readout_attn_heads'] = 4
            args['id_encoder'] = IdentifierEncoder(d_model=28, max_seq_len=500)   
            self.nnet = GraphNet(args)
        elif self.net_type == "gat_zero_id":
            args['num_channels'] = 32
            args['expand_base'] = 2
            args['attn_heads'] = 1
            args['readout_attn_heads'] = 4
            args['id_encoder'] = ZeroIdentifierEncoder(d_model=28)
            self.nnet = GraphNet(args)
        elif self.net_type == "gat_random_id":
            random_id_args(28)
            self.nnet = GraphNet(args)
        elif self.net_type == "gat_random_id_1d":
            random_id_args(1)
            self.nnet = GraphNet(args)
        elif self.net_type == "gat_random_id_10d":
            random_id_args(10)
            self.nnet = GraphNet(args)
        elif self.net_type == "gat_random_id_20d":
            random_id_args(20)
            self.nnet = GraphNet(args)
        else:
            assert False, "Unknown network {}".format(nnet)

        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()

        if args.cuda:
            self.nnet.cuda()

    # def train(self, examples, checkpoint_folder="checkpoint"):
    #     """
    #     examples: list of examples, each example is of form (board, pi, v)
    #     """
    #     optimizer = optim.Adam(self.nnet.parameters())
    #     min_loss = math.inf

    #     for epoch in range(args.epochs):
    #         print('EPOCH ::: ' + str(epoch + 1))
    #         self.nnet.train()
    #         pi_losses = AverageMeter()
    #         v_losses = AverageMeter()

    #         batch_count = int(len(examples) / args.batch_size)

    #         t = tqdm(range(batch_count), desc='Training Net')
    #         for _ in t:
    #             sample_ids = np.random.randint(len(examples), size=args.batch_size)
    #             # TODO: pad the example boards to the same size as the current training boardto give a consistent tensor size
    #             # not actions will be different on different board sizes
    #             boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))
    #             boards = torch.FloatTensor(np.array(boards).astype(np.float64))
    #             target_pis = torch.FloatTensor(np.array(pis))
    #             target_vs = torch.FloatTensor(np.array(vs).astype(np.float64))

    #             # predict
    #             if args.cuda:
    #                 boards, target_pis, target_vs = boards.contiguous().cuda(), target_pis.contiguous().cuda(), target_vs.contiguous().cuda()

    #             # compute output
    #             out_pi, out_v = self.nnet(boards)
    #             l_pi = self.loss_pi(target_pis, out_pi)
    #             l_v = self.loss_v(target_vs, out_v)
    #             total_loss = l_pi + l_v

    #             # track best model 
    #             if total_loss < min_loss:
    #                 min_loss = total_loss
    #                 self.save_checkpoint(folder=checkpoint_folder, filename='best.pth.tar')

    #             # record loss
    #             pi_losses.update(l_pi.item(), boards.size(0))
    #             v_losses.update(l_v.item(), boards.size(0))
    #             t.set_postfix(Loss_pi=pi_losses, Loss_v=v_losses)

    #             # compute gradient and do SGD step
    #             optimizer.zero_grad()
    #             total_loss.backward()
    #             optimizer.step()

    #     self.load_checkpoint(folder=checkpoint_folder, filename='best.pth.tar')

    def predict(self, board):
        """
        board: graph board
        """
        # timing
        start = time.time()

        # preparing input
        self.nnet.eval()
        with torch.no_grad():
            pi, v = self.nnet(board)

        #print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time()-start))
        return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]

    def loss_pi(self, targets, outputs):
        return -torch.sum(targets * outputs) / targets.size()[0]

    def loss_v(self, targets, outputs):
        return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        torch.save({
            'state_dict': self.nnet.state_dict(),
        }, filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise Exception("No model in path {}".format(filepath))
        map_location = None if args.cuda else 'cpu'
        checkpoint = torch.load(filepath, map_location=map_location)
        self.nnet.load_state_dict(checkpoint['state_dict'])
