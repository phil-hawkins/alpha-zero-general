import sys
sys.path.append('..')
from utils import *

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

class OpenC4NNet(nn.Module):
    def __init__(self, game, args):
        # game params
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args

        super(OpenC4NNet, self).__init__()
        self.conv1 = nn.Conv2d(1, args.num_channels, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1, padding=1)

        self.bn1 = nn.BatchNorm2d(args.num_channels)
        self.bn2 = nn.BatchNorm2d(args.num_channels)
        self.bn3 = nn.BatchNorm2d(args.num_channels)
        self.bn4 = nn.BatchNorm2d(args.num_channels)

        # 1x1 conv to 1024
        self.conv5 = nn.Conv2d(args.num_channels, 1024, 1, stride=1)
        self.bn5 = nn.BatchNorm2d(1024)

        self.fc1 = nn.Linear(1024, 512)
        self.fc_bn1 = nn.BatchNorm1d(512)

        self.fc2 = nn.Linear(512, 1)
        self.fc3 = nn.Linear(512, 1)


    def forward(self, s):
        #                                                           s: batch_size x board_x x board_y
        s = s.view(-1, 1, self.board_x, self.board_y)                # batch_size x 1 x board_x x board_y
        s = F.relu(self.bn1(self.conv1(s)))                          # batch_size x num_channels x board_x x board_y
        s = F.relu(self.bn2(self.conv2(s)))                          # batch_size x num_channels x board_x x board_y
        s = F.relu(self.bn3(self.conv3(s)))                          # batch_size x num_channels x board_x x board_y
        s = F.relu(self.bn4(self.conv4(s)))                          # batch_size x num_channels x board_x x board_y
        #s = s.view(-1, self.args.num_channels*(self.board_x-4)*(self.board_y-4))

        s = F.dropout(F.relu(self.bn5(self.conv5(s))), p=self.args.dropout, training=self.training)     # batch_size x 1024 x board_x x board_y
        s = s.mean(dim=2).permute(0, 2, 1).reshape(-1, 1024)                                               # (batch_size x board_y) x 1024
        s = F.dropout(F.relu(self.fc_bn1(self.fc1(s))), p=self.args.dropout, training=self.training)    # (batch_size x board_y) x 512

        pi = self.fc2(s).reshape(-1, self.action_size)                                                     # batch_size x action_size
        v = self.fc3(s).reshape(-1, self.action_size).mean(dim=1, keepdim=True)                            # batch_size x 1

        return F.log_softmax(pi, dim=1), torch.tanh(v)
