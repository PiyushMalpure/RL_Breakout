#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    """Initialize a deep Q-learning network
    Hints:
    -----
        Original paper for DQN
    https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
    This is just a hint. You can build your own structure.
    """

    def __init__(self):
        """
        Parameters:
        -----------
        in_channels: number of channel of input.
                i.e The number of most recent frames stacked together, here we use 4 frames, which means each state in Breakout is composed of 4 frames.
        num_actions: number of action-value to output, one-to-one correspondence to action in game.
        You can add additional arguments as you need.
        In the constructor we instantiate modules and assign them as
        member variables.
        """
        # super(DQN, self).__init__(in_channels=4, num_actions=4)
        super(DQN, self).__init__()
        self.norm = nn.LayerNorm([4,84,84], elementwise_affine=False)
        # self.conv1 = nn.Conv2d(4,6,4)
        # self.pool =nn.MaxPool2d(3,3)
        # self.conv2 = nn.Conv2d(6,16,4)
        # self.fc1 = nn.Linear(16*8*8,120)
        # self.fc2 = nn.Linear(120,64)
        # self.fc3 = nn.Linear(64,4)
        self.conv1 = nn.Conv2d(4,32, kernel_size = 8, stride = 4)
        # self.pool =nn.MaxPool2d(3,3)
        self.conv2 = nn.Conv2d(32,64,kernel_size = 4, stride = 2)
        self.conv3 = nn.Conv2d(64,64,kernel_size = 3, stride = 1)
        self.fc1 = nn.Linear(64*7*7,512)
        self.fc2 = nn.Linear(512,4)
        ###########################
        # YOUR IMPLEMENTATION HERE #

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        # print((x[0][0][0][0]))
        x = x.float()/255
       
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1,64*7*7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        ###########################
        return x