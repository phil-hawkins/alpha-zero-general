import os
import sys
import math
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
from absl import logging
sys.path.append('../../')
from utils import dotdict, AverageMeter
from NeuralNet import NeuralNet
from .models.scale_cnn import CNNHex, RecurrentCNNHex
from .models.graph_net import GraphNet
from .board_graph import IdentifierEncoder, ZeroIdentifierEncoder, RandomIdentifierEncoder, batch_to_net

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

        def base_config():
            self.args['board_size'] = game.board_size
            self.args['res_blocks'] = 5
            self.args['in_channels'] = 3

        def base_gat_config(id_encoder):
            base_config()
            self.args['num_channels'] = 32
            self.args['expand_base'] = 2
            self.args['attn_heads'] = 1
            self.args['readout_attn_heads'] = 4
            self.args['id_encoder'] = id_encoder
            self.xform_input = batch_to_net

        def base_cnn_config():
            base_config()
            self.args['num_channels'] = 128
            self.args['dropout'] = 0.3
            self.xform_input = lambda x, a, device: x.to(device)

        self.net_type = net_type
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.args = dotdict({
            'action_size': game.getActionSize()
        })

        if self.net_type == "base_cnn":
            base_cnn_config()
            self.nnet = CNNHex.base_cnn(game, self.args)
        elif self.net_type == "scalefree_base_cnn":
            base_cnn_config()
            self.nnet = CNNHex.base_cnn(game, self.args)
        elif self.net_type == "recurrent_cnn":
            base_cnn_config()
            self.args.res_blocks = 2
            self.nnet = RecurrentCNNHex.recurrent_cnn(game, self.args)
        elif self.net_type == "base_gat":
            base_gat_config(IdentifierEncoder(d_model=28, max_seq_len=500))
            self.nnet = GraphNet(self.args)
        elif self.net_type == "gat_res10":
            base_gat_config(IdentifierEncoder(d_model=28, max_seq_len=500))
            self.args['res_blocks'] = 10
            self.nnet = GraphNet(self.args)
        elif self.net_type == "gat_res15":
            base_gat_config(IdentifierEncoder(d_model=28, max_seq_len=500))
            self.args['res_blocks'] = 15
            self.nnet = GraphNet(self.args)
        elif self.net_type == "gat_res20":
            base_gat_config(IdentifierEncoder(d_model=28, max_seq_len=500))
            self.args['res_blocks'] = 20
            self.nnet = GraphNet(self.args)
        elif self.net_type == "gat_res30":
            base_gat_config(IdentifierEncoder(d_model=28, max_seq_len=500))
            self.args['res_blocks'] = 30
            self.nnet = GraphNet(self.args)
        elif self.net_type == "gat_res40":
            base_gat_config(IdentifierEncoder(d_model=28, max_seq_len=500))
            self.args['res_blocks'] = 40
            self.nnet = GraphNet(self.args)
        elif self.net_type == "gat_ch128":
            base_gat_config(IdentifierEncoder(d_model=124, max_seq_len=500))
            self.args['num_channels'] = 128
            self.nnet = GraphNet(self.args)
        elif self.net_type == "gat_zero_id":
            base_gat_config(ZeroIdentifierEncoder(d_model=28))
            self.nnet = GraphNet(self.args)
        elif self.net_type == "gat_random_id":
            base_gat_config(RandomIdentifierEncoder(d_model=28))
            self.nnet = GraphNet(self.args)
        elif self.net_type == "gat_random_id_1d":
            base_gat_config(RandomIdentifierEncoder(d_model=1))
            self.nnet = GraphNet(self.args)
        elif self.net_type == "gat_random_id_10d":
            base_gat_config(RandomIdentifierEncoder(d_model=10))
            self.nnet = GraphNet(self.args)
        elif self.net_type == "gat_random_id_20d":
            base_gat_config(RandomIdentifierEncoder(d_model=20))
            self.nnet = GraphNet(self.args)
        else:
            raise Exception("Unknown model type {}".format(net_type))

        self.action_size = game.getActionSize()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.nnet.to(device=self.device)

    def train(self, examples, checkpoint_folder="checkpoint", summary_writers=None):
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
            out_pi, out_v = self.nnet(self.xform_input(boards, self.args, self.device))
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
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)
        min_loss = math.inf

        for epoch in range(self.epochs):
            logging.info('EPOCH ::: {}'.format(epoch + 1))
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

            if summary_writers is not None:
                summary_writers['train'].add_scalar("loss/policy", pi_losses.avg, global_step=epoch)
                summary_writers['train'].add_scalar("loss/value", v_losses.avg, global_step=epoch)
                summary_writers['train'].add_scalar("loss/all", v_losses.avg + pi_losses.avg, global_step=epoch)
                summary_writers['train'].add_scalar("lr", optimizer.param_groups[0]['lr'], global_step=epoch)
                summary_writers['train'].flush()

            self.nnet.eval()
            pi_losses = AverageMeter()
            v_losses = AverageMeter()

            with torch.no_grad():
                t = tqdm(range(val_batch_count), desc='Validating Net')
                for i in t:
                    batch_start = (train_batch_count + i) * self.batch_size
                    batch_end = batch_start + self.batch_size
                    step(batch_start, batch_end)

            if summary_writers is not None:
                summary_writers['val'].add_scalar("loss/policy", pi_losses.avg, global_step=epoch)
                summary_writers['val'].add_scalar("loss/value", v_losses.avg, global_step=epoch)
                summary_writers['val'].add_scalar("loss/all", v_losses.avg + pi_losses.avg, global_step=epoch)
                summary_writers['val'].flush()

            # track best model
            total_loss = pi_losses.avg + v_losses.avg
            scheduler.step(total_loss)
            if total_loss < min_loss:
                logging.info('Best loss so far! Saving checkpoint.')
                min_loss = total_loss
                self.save_checkpoint(folder=checkpoint_folder, filename=self.net_type+'_best.pth.tar')

        self.load_checkpoint(folder=checkpoint_folder, filename=self.net_type+'_best.pth.tar')

    def predict(self, board):
        """
        board: np array with board or graph board
        """
        # timing
        # start = time.time()

        self.nnet.eval()
        with torch.no_grad():
            pi, v = self.nnet(self.xform_input(board, self.args, self.device))

        return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]

    def process(self, batch):
        batch = batch.to(device=self.device)
        self.nnet.eval()
        with torch.no_grad():
            pi, v = self.nnet(self.xform_input(batch, self.args, self.device))

        return torch.exp(pi), v

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
