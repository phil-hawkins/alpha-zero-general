"""
The basic CNN hex model as per https://webdocs.cs.ualberta.ca/~hayward/papers/movepredhex.pdf 
"""

import numpy as np
import torch
from collections import OrderedDict
from torch.nn import Sequential, Linear, ReLU, Module, LogSoftmax, Conv2d, BatchNorm2d, Flatten, Tanh
import sys


class PaddHexBoard(torch.nn.Module):
    def __init__(self, padding):
        super(PaddHexBoard, self).__init__()
        self.padding = padding

    def forward(self, x):
        x = torch.nn.functional.pad(x, (self.padding, self.padding, self.padding, self.padding))
        # white has the top and bottom border
        x[:,0,:self.padding,:] = 1.0
        x[:,0,-self.padding:,:] = 1.0
        # black has the left and right border
        x[:,1,:,:self.padding] = 1.0
        x[:,1,:,-self.padding:] = 1.0
        
        return x

def multiplane(x):
    # plane index 0/1/2 - black/white/empty
    planes = []
    for cell_content in [-1, 1, 0]:
        planes.append(x == cell_content)

    return torch.stack(planes, dim=1).float()

class CNNHex(Module):

    def __init__(self, game, args):
    #def __init__(self, layers=8, filters=128, in_channels=5, size=13, position_bias=True):
        ''' reimplementation of the MoHex-CNN model from https://webdocs.cs.ualberta.ca/~hayward/papers/movepredhex.pdf 

        this model only uses stone position features (planes 0/1/2) and not the engineered bridge or "to play" features

        the position bias step has been removed
        '''

        super(CNNHex, self).__init__()
        filters = args.num_channels
        layers = args.receptive_range - 2
        assert(layers > 0)
        self.in_channels = args.in_channels

        self.module_odict = [
            ('trunk pad 0', PaddHexBoard(2)),
            ('trunk conv 0', Conv2d(in_channels=self.in_channels, out_channels=filters, kernel_size=5, padding=0)), 
            ('trunk bn 0', BatchNorm2d(num_features=filters)),
            ('trunk relu 0', ReLU())
        ]

        for i in range(1, layers):
            self.module_odict.append(('trunk pad {}'.format(i), PaddHexBoard(1)))
            self.module_odict.append(('trunk conv {}'.format(i), Conv2d(in_channels=filters, out_channels=filters, kernel_size=3, padding=0)))
            self.module_odict.append(('trunk bn {}'.format(i), BatchNorm2d(num_features=filters)))
            self.module_odict.append(('trunk relu {}'.format(i), ReLU()))

        self.trunk = Sequential(OrderedDict(self.module_odict))

        self.module_odict = [
            #('p_head cnn 0', Conv2d(in_channels=filters, out_channels=filters, kernel_size=1)),
            #('p_head bn 0', , BatchNorm2d(num_features=filters)),
            #('p_head relu 0', ReLU()),
            ('p_head cnn'.format(i), Conv2d(in_channels=filters, out_channels=1, kernel_size=1)),
            ('p_head flatten', Flatten()),
            ('p_head softmax', LogSoftmax(dim=1))
        ]
            
        self.p_head = Sequential(OrderedDict(self.module_odict))

        self.module_odict = [
            ('v_head cnn', Conv2d(in_channels=filters, out_channels=1, kernel_size=1)),
            ('v_head flatten', Flatten()),
            ('v_head linear', Linear(in_features=args.board_size**2, out_features=1)),
            ('v_head tanh', Tanh())
        ]
            
        self.v_head = Sequential(OrderedDict(self.module_odict))


    def forward(self, x):
        x = multiplane(x)
        assert(x.size(1) == self.in_channels)

        x = self.trunk(x)
        p = self.p_head(x)
        v = self.v_head(x)
            
        return p, v
