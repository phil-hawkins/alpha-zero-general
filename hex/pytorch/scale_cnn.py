"""
The basic CNN hex model based on
- https://webdocs.cs.ualberta.ca/~hayward/papers/movepredhex.pdf 
- http://webdocs.cs.ualberta.ca/~hayward/papers/3hnn.pdf

This is modified to be fully convolutional in order to accept any scale of board.
The value head uses a simple mean rather than a fixed size logistic layer in order to be scale free

"""

import numpy as np
import torch
from collections import OrderedDict, namedtuple
from torch.nn import Sequential, Linear, ReLU, Module, LogSoftmax, Conv2d, BatchNorm2d, Flatten, Tanh
import sys
from utils import dotdict


class ResBlock(torch.nn.Module):
    def __init__(self, channels=32, affine_bn=True):
        super(ResBlock, self).__init__()
        self.cnn_1 = Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)
        self.bn_1 = BatchNorm2d(num_features=channels, affine=affine_bn)
        self.cnn_2 = Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)
        self.bn_2 = BatchNorm2d(num_features=channels, affine=affine_bn)

    def forward(self, x):
        residual = x
        x = self.cnn_1(x)
        x = self.bn_1(x).relu()
        x = self.cnn_2(x)
        x = self.bn_2(x).relu()
        x += residual
        
        return x.relu()

class ValueHead(torch.nn.Module):
    def __init__(self, in_channels, feature_size):
        super(ValueHead, self).__init__()
        self.v_head = Sequential(OrderedDict([
            ('v_head cnn', Conv2d(in_channels=in_channels, out_channels=1, kernel_size=1)),
            ('v_head flatten', Flatten()),
            ('v_head linear', Linear(in_features=feature_size, out_features=1)),
            ('v_head tanh', Tanh())
        ]))

    def forward(self, x):
        x = self.v_head(x)
        
        return x

class ScaleFreeValueHead(torch.nn.Module):
    def __init__(self, in_channels):
        super(ScaleFreeValueHead, self).__init__()
        self.v_head = Sequential(OrderedDict([
            ('v_head cnn', Conv2d(in_channels=in_channels, out_channels=1, kernel_size=1)),
            ('v_head flatten', Flatten())
        ]))

    def forward(self, x):
        x = self.v_head(x).mean(dim=1)
        
        return x


def multiplane(x):
    # plane index 0/1/2 - black/white/empty
    planes = []
    for cell_content in [-1, 1, 0]:
        planes.append(x == cell_content)

    return torch.stack(planes, dim=1).float()

class CNNHex(Module):

    def __init__(self, game, args, v_head):
    #def __init__(self, layers=8, filters=128, in_channels=5, size=13, position_bias=True):
        ''' reimplementation of the MoHex-CNN model from https://webdocs.cs.ualberta.ca/~hayward/papers/movepredhex.pdf 

        this model only uses stone position features (planes 0/1/2) and not the engineered bridge or "to play" features

        the position bias step has been removed
        '''

        super(CNNHex, self).__init__()
        self.in_channels = args.in_channels

        self.trunk_odict = [
            ('trunk conv 0', Conv2d(in_channels=self.in_channels, out_channels=args.num_channels, kernel_size=3, padding=1)), 
            ('trunk bn 0', BatchNorm2d(num_features=args.num_channels)),
            ('trunk relu 0', ReLU())
        ]
        for i in range(args.res_blocks):
            self.trunk_odict.append(('res block {}'.format(i), ResBlock(channels=args.num_channels)))           
        self.trunk = Sequential(OrderedDict(self.trunk_odict))

        self.p_head = Sequential(OrderedDict([
            ('p_head cnn', Conv2d(in_channels=args.num_channels, out_channels=1, kernel_size=1)),
            ('p_head flatten', Flatten()),
            ('p_head softmax', LogSoftmax(dim=1))
        ]))

        self.v_head = v_head

    @classmethod
    def base_cnn(cls, game, args):
        return cls(game, args, ValueHead(in_channels=args.num_channels, feature_size=args.board_size**2))

    @classmethod
    def scalefree_base_cnn(cls, game, args):
        return cls(game, args, ScaleFreeValueHead(in_channels=args.num_channels))


    def forward(self, x):
        x = multiplane(x)
        assert(x.size(1) == self.in_channels)

        x = self.trunk(x)
        p = self.p_head(x)
        v = self.v_head(x)
            
        return p, v

class RecurrentCNNHex(CNNHex):

    def __init__(self, game, args, v_head):
    #def __init__(self, layers=8, filters=128, in_channels=5, size=13, position_bias=True):
        ''' reimplementation of the MoHex-CNN model from https://webdocs.cs.ualberta.ca/~hayward/papers/movepredhex.pdf 

        this model only uses stone position features (planes 0/1/2) and not the engineered bridge or "to play" features

        the position bias step has been removed
        '''

        super(RecurrentCNNHex, self).__init__(game, args, v_head)
        self.msg_passing = ResBlock(channels=args.num_channels, affine_bn=False)

    @classmethod
    def recurrent_cnn(cls, game, args):
        return cls(game, args, ScaleFreeValueHead(in_channels=args.num_channels))

    def forward(self, x):
        x = multiplane(x)
        assert(x.size(1) == self.in_channels)
        board_size = x.size(2)

        x = self.trunk(x)
        for _ in range(board_size):
            x = self.msg_passing(x)

        p = self.p_head(x)
        v = self.v_head(x)
            
        return p, v