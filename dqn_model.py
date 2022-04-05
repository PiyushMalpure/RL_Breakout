#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

