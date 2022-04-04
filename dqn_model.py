# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# import torch.nn as nn
# import torch.nn.functional as F
# import torch


# class DQN(nn.Module):
#     """Initialize a deep Q-learning network

#     Hints:
#     -----
#         Original paper for DQN
#     https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf

#     This is just a hint. You can build your own structure.
#     """

#     def __init__(self, in_channels=4, num_actions=4):
#         """
#         Parameters:
#         -----------
#         in_channels: number of channel of input.
#                 i.e The number of most recent frames stacked together, here we use 4 frames, which means each state in Breakout is composed of 4 frames.
#         num_actions: number of action-value to output, one-to-one correspondence to action in game.

#         You can add additional arguments as you need.
#         In the constructor we instantiate modules and assign them as
#         member variables.
#         """
#         super(DQN, self).__init__()
#         ###########################
#         # YOUR IMPLEMENTATION HERE #
#         self.cnn = nn.Sequential(nn.Conv2d(4, 16, kernel_size = 8, stride = 4),
#                                     nn.ReLU(True),
#                                     nn.Conv2d(16, 32, kernal_size=4, stride = 2),
#                                     nn.ReLU(True)
#                                     )
#         self.classifier = nn.Sequential(nn.Linear(9*9*32, 256),
#                                         nn.ReLU(True),
#                                         nn.Linear(256, 1))

#     def forward(self, x):
#         """
#         In the forward function we accept a Tensor of input data and we must return
#         a Tensor of output data. We can use Modules defined in the constructor as
#         well as arbitrary operators on Tensors.
#         """
#         ###########################
#         # YOUR IMPLEMENTATION HERE #
#         x = self.cnn(x)
#         x = torch.flatten(x, start_dim = 1)
#         x = self.classifier(x)
#         ###########################
#         return x

import torch
import torch.nn as nn
import torch.nn.functional as F


############################ Convolution Model ############################

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet,self).__init__()
        self.conv1 = nn.Conv2d(4,32, kernel_size = 8, stride = 4)
        self.conv2 = nn.Conv2d(32,64,kernel_size = 4, stride = 2)
        self.conv3 = nn.Conv2d(64,64,kernel_size = 3, stride = 1)
    def forward(self,x):
        x = x.float()/255
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1,64*7*7)
        return x

############################ Dueling DQN Model  ############################

class DuelNet(nn.Module):
    def __init__(self):
        super(DuelNet,self).__init__()
        self.fc = nn.Linear(64*7*7,1024)
        self.val = nn.Linear(512,1)
        self.adv = nn.Linear(512,4)
    def forward(self,x):
        x = F.relu(self.fc(x))
        val,adv = torch.split(x,512,dim=1)
        val = self.val(val)
        adv = self.adv(adv)
        x = val + adv - torch.mean(adv,dim=1,keepdim=True)
        return x

############################ Normal DQN Model  ############################

class LinNet(nn.Module):
    def __init__(self):
        super(LinNet,self).__init__()
        self.fc1 = nn.Linear(64*7*7,512)
        self.fc2 = nn.Linear(512,4)
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

############################ Bootstrap DQN Model  ############################

class BootNet(nn.Module):
    def __init__(self,n_heads, duel = False):
        super(BootNet,self).__init__()
        self.conv_net = ConvNet()
        if duel:
            print("Time for a DuelDQN!")
            self.network_list = nn.ModuleList([DuelNet() for head in range(n_heads)])
        else:
            print("Time for a DQN!")
            self.network_list = nn.ModuleList([LinNet() for head in range(n_heads)])

    def _conv(self,x):
        return self.conv_net(x)
    
    def _lin(self,x):
        return [lin(x) for lin in self.network_list]

    def forward(self,x,head_index=None):
        if head_index is not None:
            return self.network_list[head_index](self._conv(x))
        else:
            conv = self._conv(x)
            return self._lin(conv)

