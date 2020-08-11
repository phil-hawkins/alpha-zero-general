import os
import sys
import time
import math

import numpy as np
from tqdm import tqdm

sys.path.append('../../')
from utils import *
from NeuralNet import NeuralNet

import torch
import torch.optim as optim

#from .Connect4NNet import Connect4NNet as c4nnet
from .models.scale_cnn import CNNHex, RecurrentCNNHex
from .models.graph_net import GraphNet, NativeGraphNet
from .board_graph import Board, BoardGraph, PlayerGraph, IdentifierEncoder, ZeroIdentifierEncoder, RandomIdentifierEncoder, batch_to_net
from .matrix_hex_board import MatrixHexBoard
from .graph_hex_board import GraphHexBoard

# args = dotdict({
#     'dropout': 0.3,
#     'num_channels': 128,#32,
#     'res_blocks' : 5,
#     'in_channels' : 3                   # 0/1/2 - black/white/empty
# })


# class FakeNNet(NeuralNet):
#     """ fake neural network that does random predictions for pitting an oponent against pure MCTS
#     """
#     def __init__(self, game, net_type=None, value_function=None):
#         self.game = game
#         self.value_function = value_function if value_function else lambda x : 0.

#     def predict(self, board):
#         valids = self.game.getValidMoves(board)
#         pi = np.zeros_like(valids, dtype=np.float32)
#         valids_ndx = np.nonzero(valids)[0]
#         np.random.shuffle(valids_ndx)
#         action_ndx = valids_ndx[0]
#         pi[action_ndx] = 1.0
#         v = self.value_function(board)

#         return pi, v


# def value_from_shortest_path(board):
#     """ takes either a matrix representation of the board or a graph representation
#     and calculates a state value based on a comparison of shortest paths of each player
#     """
#     if type(board).__module__ == np.__name__:
#         bg = BoardGraph.from_matrix_board(MatrixHexBoard(torch.tensor(board)))
#     elif isinstance(board, GraphHexBoard):
#         bg = BoardGraph.from_graph_board(board)
#     else:
#         raise Exception("Unsupported board type")

#     g_p1, g_p2 = PlayerGraph.from_board_graph(bg, 1), PlayerGraph.from_board_graph(bg, -1)
#     sp_p1, sp_p2 = g_p1.shortest_path(), g_p2.shortest_path()

#     if sp_p1 == 0:
#         v = 1.0
#     elif sp_p2 == 0:
#         v = -1.0
#     else:
#         v = (sp_p2 - sp_p1) / max(sp_p1, sp_p2)

#     return v

class NNetWrapper(NeuralNet):
    def __init__(self, game, net_type="base_gat", lr=1e-3, epochs=10, batch_size=64):

        def base_gat_args(id_encoder):
            self.args['num_channels'] = 32
            self.args['expand_base'] = 2
            self.args['attn_heads'] = 1
            self.args['readout_attn_heads'] = 4
            self.args['id_encoder'] = id_encoder

        self.net_type = net_type
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.args = dotdict({
            'action_size': game.getActionSize(),
            'dropout': 0.3,
            'num_channels': 128,
            'res_blocks' : 5,
            'in_channels' : 3                   # 0/1/2 - black/white/empty            
        })

        if self.net_type == "base_cnn":
            self.nnet = CNNHex.base_cnn(game, self.args)
        elif self.net_type == "scalefree_base_cnn":
            self.nnet = CNNHex.base_cnn(game, self.args)
        elif self.net_type == "recurrent_cnn":
            self.args.res_blocks = 2
            self.nnet = RecurrentCNNHex.recurrent_cnn(game, self.args)
        elif self.net_type == "base_gat":
            base_gat_args(IdentifierEncoder(d_model=28, max_seq_len=500))
            self.nnet = GraphNet(self.args)
        elif self.net_type == "gat_res10":
            base_gat_args(IdentifierEncoder(d_model=28, max_seq_len=500))
            self.args['res_blocks'] = 10
            self.nnet = GraphNet(self.args)
        elif self.net_type == "gat_zero_id":
            base_gat_args(ZeroIdentifierEncoder(d_model=28))
            self.nnet = GraphNet(self.args)
        elif self.net_type == "gat_random_id":
            random_id_args(RandomIdentifierEncoder(d_model=28))
            self.nnet = GraphNet(self.args)
        elif self.net_type == "gat_random_id_1d":
            random_id_args(RandomIdentifierEncoder(d_model=1))
            self.nnet = GraphNet(self.args)
        elif self.net_type == "gat_random_id_10d":
            random_id_args(RandomIdentifierEncoder(d_model=10))
            self.nnet = GraphNet(self.args)
        elif self.net_type == "gat_random_id_20d":
            random_id_args(RandomIdentifierEncoder(d_model=20))
            self.nnet = GraphNet(self.args)
        else:
            raise Exception("Unknown model type {}".format(nnet))

        self.action_size = game.getActionSize()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.nnet.to(device=self.device)

    def train(self, examples, checkpoint_folder="checkpoint"):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """

        def prep(t):
            return t.contiguous().to(device=self.device)

        def step(batch_start, batch_end):
            boards, pis, vs = list(zip(*examples[batch_start:batch_end]))

            boards = torch.FloatTensor(np.array(boards).astype(np.float64))
            target_pis = torch.FloatTensor(np.array(pis))
            target_vs = torch.FloatTensor(np.array(vs).astype(np.float64))

            boards, target_pis, target_vs = prep(boards), prep(target_pis), prep(target_vs)

            # compute output
            out_pi, out_v = self.nnet(batch_to_net(boards, self.args, self.device))
            l_pi = self.loss_pi(target_pis, out_pi)
            l_v = self.loss_v(target_vs, out_v)
            total_loss = l_pi + l_v

            # record loss
            pi_losses.update(l_pi.item(), boards.size(0))
            v_losses.update(l_v.item(), boards.size(0))
            t.set_postfix(Loss_pi=pi_losses, Loss_v=v_losses)

            return total_loss
            
        # TODO: add support for graph based board representation
        optimizer = optim.Adam(self.nnet.parameters(), lr=self.lr)
        min_loss = math.inf

        for epoch in range(self.epochs):
            print('EPOCH ::: ' + str(epoch + 1))
            self.nnet.train()
            pi_losses = AverageMeter()
            v_losses = AverageMeter()

            batch_count = int(len(examples) / self.batch_size)
            train_batch_count = int(batch_count * .9)
            val_batch_count = batch_count - train_batch_count

            t = tqdm(range(train_batch_count), desc='Training Net')
            for i in t:
                batch_start = i * self.batch_size
                batch_end = batch_start + self.batch_size
                total_loss = step(batch_start, batch_end)

                # compute gradient and do SGD step
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
            
            
            self.nnet.eval()
            pi_losses = AverageMeter()
            v_losses = AverageMeter()

            with torch.no_grad():
                t = tqdm(range(val_batch_count), desc='Validating Net')
                for i in t:
                    batch_start = (train_batch_count + i) * self.batch_size
                    batch_end = batch_start + self.batch_size
                    step(batch_start, batch_end)

            # track best model
            total_loss = pi_losses.avg + v_losses.avg
            if total_loss < min_loss:             
                print('Best loss so far! Saving checkpoint.')
                min_loss = total_loss
                self.save_checkpoint(folder=checkpoint_folder, filename='best.pth.tar')  

        self.load_checkpoint(folder=checkpoint_folder, filename='best.pth.tar')

    def predict(self, board):
        """
        board: np array with board or graph board
        """
        # timing
        # start = time.time()

        self.nnet.eval()
        with torch.no_grad():
            pi, v = self.nnet(batch_to_net(board, self.args, self.device))

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
        # else:
        #     print("Checkpoint Directory exists! ")
        torch.save({
            'state_dict': self.nnet.state_dict(),
        }, filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise Exception("No model in path {}".format(filepath))
        
        checkpoint = torch.load(filepath, map_location=self.device)
        self.nnet.load_state_dict(checkpoint['state_dict'])
