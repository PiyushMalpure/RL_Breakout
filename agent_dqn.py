#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
import numpy as np
from collections import deque
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from agent import Agent
from dqn_model import DQN
"""
you can import any package and define any extra function as you need
"""

torch.manual_seed(595)
np.random.seed(595)
random.seed(595)

class my_dataset(Dataset):
    def __init__(self,data):
        self.samples = data
    def __len__(self):
        return len(self.samples)
    def __getitem__(self,idx):
        return self.samples[idx]

class Agent_DQN(Agent):
    def __init__(self, env, args):
        """
        Initialize everything you need here.
        For example: 
            paramters for neural network  
            initialize Q net and target Q net
            parameters for repaly buffer
            parameters for q-learning; decaying epsilon-greedy
            ...
        """

        super(Agent_DQN,self).__init__(env)
        ###########################
        # YOUR IMPLEMENTATION HERE #
        self.epochs = 10
        self.n_episodes = 1000000
        self.env = env
        self.nA = self.env.action_space.n
        # self.nS = self.env.observation_space
        self.batch_size = 32
        self.DQN = DQN()
        self.Target_DQN = DQN()
        self.buffer_memory = 1000000
        self.train_buffer_size = 4
        self.min_buffer_size = 10000
        self.target_update_buffer = 10000
        self.learning_rate = 0.0001
        self.discount_factor = 0.999
        self.epsilon = 1
        self.min_epsilon = 0.01
        # self.decay_rate = 0.999
        self.ep_decrement = (self.epsilon - self.min_epsilon)/self.n_episodes
        self.criteria = nn.MSELoss()
        self.optimiser = optim.Adam(self.DQN.parameters(),self.learning_rate)
        self.buffer=[]
        self.Evaluation = 100000
        self.total_evaluation__episodes = 100
        self.full_train = 100000
        if args.test_dqn:
            #you can load your model here
            print('loading trained model')
            ###########################
            # YOUR IMPLEMENTATION HERE #
            

    def init_game_setting(self):
        """
        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary.
        If no parameters need to be initialized, you can leave it as blank.
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        obs = self.env.reset()
        ###########################
        pass
    
    
    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent
        Input:
            observation: np.array
                stack 4 last preprocessed frames, shape: (84, 84, 4)
        Return:
            action: int
                the predicted action from trained model
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        if not test:
            p = random.random()
            if p < self.epsilon:
                action = np.random.randint(0,self.nA)
            else:
                a = self.DQN(torch.from_numpy(observation).unsqueeze(0))
                action = np.argmax(a.detach().numpy())
        else:
            a = self.Target_DQN(torch.from_numpy(observation).unsqueeze(0))
            action = np.argmax(a.detach().numpy())
        ###########################
        return action
    
    def push(self,episode):
        """ You can add additional arguments as you need. 
        Push new data to buffer and remove the old one if the buffer is full.
        
        Hints:
        -----
            you can consider deque(maxlen = 10000) list
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        if len(self.buffer) < self.buffer_memory:
            self.buffer.append(episode)
        else:
            self.buffer.pop(0)
            self.buffer.append(episode)
        ###########################
        
        
    def replay_buffer(self):
        """ You can add additional arguments as you need.
        Select batch from buffer.
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        
        batch = random.sample(self.buffer,self.batch_size)
        # print(np.shape(batch[0][:]))
        batch = list(zip(*batch))
        # print(np.asarray(batch[1]))
        batch_x = torch.from_numpy(np.asarray(batch[0]))
        act = torch.from_numpy(np.vstack(batch[1]))
        rew = torch.from_numpy(np.asarray(batch[2]))
        dones = torch.from_numpy(np.asarray(batch[3]))
        batch_y = torch.from_numpy(np.asarray(batch[4]))
        # print(act.shape)
        ###########################
        return batch_x,act,rew,dones,batch_y
        

    def train(self):
        """
        Implement your training algorithm here
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        current = 1
        reward_list =[]
        loss_list= []
        current_train = 1
        current_target = 1
        for x in range(self.n_episodes):
            obs = np.transpose(self.env.reset(),(2,0,1))
            # print(obs[0][40][:])
            e_list=[]
            done = False
            accumulated_rewards = 0
            while not done:
                # self.env.render()
                action = self.make_action(obs,False)
                next_obs,reward,done,info = self.env.step(action)
                next_obs = np.transpose(next_obs,(2,0,1))
                # print(info['ale.lives'])
                # print(np.shape(e_list[-1]))
                accumulated_rewards+=reward
                self.push([obs,action,reward,done,next_obs])
                self.epsilon-=self.ep_decrement
                if current_train % self.train_buffer_size == 0 and len(self.buffer) > self.min_buffer_size:
                    batch_x,act,rew,dones,batch_y = self.replay_buffer()
                    self.optimiser.zero_grad()
                    future_return =  self.Target_DQN(batch_y).max(1)[0].detach() * self.discount_factor
                    future_return[dones] = 0
                    y = rew + future_return
                    c_q = self.DQN(batch_x).gather(1,act)
                    loss = self.criteria(c_q.double(),(y.double()).unsqueeze(1))
                    loss_list.append(loss.detach())
                    loss.backward()
                    # self.env.render()
                    self.optimiser.step()
                    current_train = 1

                if current_target > self.target_update_buffer:
                    self.Target_DQN.load_state_dict(self.DQN.state_dict())
                    current_target = 1

                if current % self.full_train == 0:
                    # current = 1
                    # print("\n Weights: \n",list(self.DQN.parameters()),"\r")
                    dataset = my_dataset(self.buffer)
                    for i in range(self.epochs):
                        loader = torch.utils.data.DataLoader(dataset, batch_size = 32, shuffle = True)
                        # print(len(list(loader)))
                        for batch in list(loader):
                            batch_x,act,rew,dones,batch_y=batch
                            self.optimiser.zero_grad()
                            future_return =  self.Target_DQN(batch_y).max(1)[0].detach() * self.discount_factor
                            future_return[dones] = 0
                            y = rew + future_return
                            c_q = self.DQN(batch_x).gather(1,act.unsqueeze(1))
                            loss = self.criteria(c_q.double(),y.double().unsqueeze(1))
                            loss_list.append(loss.detach())
                            loss.backward()
                            self.optimiser.step()
                
                if current % self.Evaluation == 0:
                    # current = 1
                    # print("\n Weights: \n",list(self.DQN.parameters()),"\r")
                    print("\n","#" * 40, "Evaluation number %d"%(current/self.Evaluation),"#" * 40)
                    for i in range(self.total_evaluation__episodes):
                        state = np.transpose(self.env.reset(),(2,0,1))
                        done = False
                        episode_reward = 0.0
                        rewards=[]
                        #playing one game
                        while(not done):
                            action = self.make_action(state, test=True)
                            state, reward, done, info = self.env.step(action)
                            episode_reward += reward
                            state = np.transpose(state,(2,0,1))
                        rewards.append(episode_reward)
                    print('Run %d episodes'%(self.total_evaluation__episodes))
                    print('Mean:', np.mean(rewards))
                    print("#" * 40, "Evaluation Ended!","#" * 40,"\n")


                current+=1
                current_train += 1
                current_target += 1
                obs = next_obs
            reward_list.append(accumulated_rewards)
            if len(reward_list) % 200 == 0:
                reward_list = reward_list[-150:]
                # print(reward_list)
                loss_list = loss_list[-150:]
            if x%100 == 0:
                print("Current = %d, episode = %d, Average_reward = %0.2f, epsilon = %0.2f"%(current, x+1, np.mean(reward_list[-100:]), self.epsilon))






        
        ###########################