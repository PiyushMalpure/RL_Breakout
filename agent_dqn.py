#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
import numpy as np
from collections import deque,Counter
import os
import sys

from environment import Environment
from test import test
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from agent import Agent
from dqn_model import BootNet
from torch.utils.tensorboard import SummaryWriter
"""
you can import any package and define any extra function as you need
"""

torch.manual_seed(595)
np.random.seed(595)
random.seed(595)
Path_weights = './last_train_weights_bootdqn_3.tar'
Path_memory = './last_memory_bootdqn.tar'
tensor_board_dir='./logs/train_data'
writer = SummaryWriter(tensor_board_dir)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ",device)



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
        
        ############# Train Parameters #############
        
        self.epochs = 10
        self.args = args
        self.n_episodes = 10000000
        self.env = env
        self.nA = self.env.action_space.n
        self.batch_size = 32
        self.eval_num=0
        self.n_heads = int(self.args.n_heads)
        self.learning_rate = 0.0000625
        self.discount_factor = 0.99
        self.Evaluation = 100000
        self.total_evaluation__episodes = 100
        self.full_train = 100000
        
        ############# Model Parameters #############
        self.Duel_DQN = True
        self.Double_DQN = False
        self.DQN = BootNet(self.n_heads,self.Duel_DQN).to(device)
        self.Target_DQN = BootNet(self.n_heads,self.Duel_DQN).to(device)
        self.criteria = nn.SmoothL1Loss()
        self.optimiser = optim.Adam(self.DQN.parameters(),self.learning_rate)

        ############# Buffer Parameters #############
        
        self.buffer_memory = 500000
        self.train_buffer_size = 4
        self.min_buffer_size = 50000
        self.target_update_buffer = 10000
        self.buffer=[]
        
        ############# Epsilon Parameters #############
        
        self.max_steps = 25000000
        self.annealing_steps = 1000000
        self.start_epsilon = 1
        self.end_epsilon_1 = 0.1
        self.end_epsilon_2 = 0.01
        self.slope1 = -(self.start_epsilon - self.end_epsilon_1)/self.annealing_steps
        self.constant1 = self.start_epsilon - self.slope1*self.min_buffer_size
        self.slope2 = -(self.end_epsilon_1 - self.end_epsilon_2)/(self.max_steps - self.annealing_steps - self.min_buffer_size)
        self.constant2 = self.end_epsilon_2 - self.slope2*self.max_steps

        ############# Other Train Parameters #############
        
        self.next_obs = np.transpose(self.env.reset(),(2,0,1))
        self.done = False
        self.terminal = False
        self.x = 0
        self.ep = 0
        self.current = 0
        self.reward_list =[]
        self.loss_list= []
        self.current_train = 0
        self.current_target = 0
        self.max_test_reward=321
        self.head_list = list(range(self.n_heads))

        writer.add_hparams({"Learning_Rate":self.learning_rate,"Batch_Size":self.batch_size,"Discount Factor":self.discount_factor,"Total Episodes":self.n_episodes,"Buffer Size":self.buffer_memory},{"Max__Test_Reward":self.max_test_reward})
        
        ############# Continue Training #############
        
        if args.cont:
          print("#"*50+"Resuming Training"+"#"*50)
          dic_weights = torch.load(Path_weights,map_location=device)
          dic_memory = torch.load(Path_memory)
          self.epsilon = dic_memory['epsilon']
          #self.epsilon = 0.0001
          self.x = dic_memory['x']
          self.ep = dic_memory['ep']
          print(self.ep)
          # self.ep_decrement = (1 - self.min_epsilon)/(self.n_episodes)
          self.current = dic_memory['current_info'][0]
          self.current_target = dic_memory['current_info'][1]
          self.current_train = dic_memory['current_info'][2]
          self.next_obs = dic_memory['next_info'][0]
          self.done = dic_memory['next_info'][1]
          self.terminal = dic_memory['next_info'][2]
          self.reward_list = []
          self.DQN.load_state_dict(dic_weights['train_state_dict'])
          self.Target_DQN.load_state_dict(dic_weights['target_state_dict'])
          self.DQN.train()
          self.Target_DQN.train()
          self.optimiser.load_state_dict(dic_weights['optimiser_state_dict'])
        
        ############# Testing #############
        
        if args.test_dqn:
            #you can load your model here
            print('loading trained model')
            ###########################
            # YOUR IMPLEMENTATION HERE #
            dic_weights = torch.load(Path_weights,map_location=device)
            torch.save(dic_weights['train_state_dict'],'trained_model.pth')
            # self.DQN.load_state_dict(dic_weights['train_state_dict'])
            # self.DQN.eval()
            

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
    
    
    def make_action(self, observation, active_head = None, test=True):
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
        
        if test:
            self.epsilon = 0
        
        elif self.current < self.min_buffer_size:
            self.epsilon = 1
        
        elif self.current >= self.min_buffer_size and self.current < self.min_buffer_size + self.annealing_steps:
            self.epsilon = self.current*self.slope1 + self.constant1
        
        elif self.current >= self.min_buffer_size + self.annealing_steps:
            self.epsilon = self.current*self.slope2 + self.constant2
        
        else:
            self.epsilon = 0

        p = np.random.rand()
        if p < self.epsilon:
            action = np.random.randint(0,self.nA)
        else:
            q_values = self.DQN(torch.from_numpy(observation).unsqueeze(0).to(device),active_head)
            if active_head is not None:
                action = torch.argmax(q_values.data,dim = 1).item()
            else:
                acts = [torch.argmax(q_values[k].data,dim = 1).item() for k in range(self.n_heads)]
                data = Counter(acts)
                action = data.most_common(1)[0][0]
                # print(action)

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
        act = torch.from_numpy(np.asarray(batch[1]))
        rew = torch.from_numpy(np.asarray(batch[2]))
        dones = torch.from_numpy(np.asarray(batch[3])).to(device)
        batch_y = torch.from_numpy(np.asarray(batch[4]))
        mask = torch.from_numpy(np.asarray(batch[5])).to(device)
        # print(act.shape)
        ###########################
        return batch_x,act,rew,dones,batch_y,mask
        

    def learn(self):
        
        self.optimiser.zero_grad()
        
        batch_x,actions,rew,dones,batch_y,masks = self.replay_buffer()
        # print(' ',batch_x.size(),'\n',actions.size(),'\n',rew.size(),'\n',dones.size(),'\n',batch_y.size(),'\n',masks.size(),'\n')
        Predicted_q_vals_list = self.DQN(batch_x.to(device))
        Target_q_vals_list = self.Target_DQN(batch_y.to(device))
        Target_policy_vals_list = self.DQN(batch_y.to(device))
        # print('',len(Predicted_q_vals_list),'\n',len(Target_q_vals_list),'\n',len(Target_policy_vals_list),'\n')
        count_losses = []
        for k in range(self.n_heads):
            
            total_used = torch.sum(masks[:,k])
            
            if total_used > 0:
                Target_q_values = Target_q_vals_list[k].data
                
                if (self.Double_DQN):
                    next_actions = Target_policy_vals_list[k].data.max(1,True)[1]
                    Y = Target_q_values.gather(1,next_actions).squeeze(1)
                else:
                    Y = Target_q_values.max(1,True)[0].squeeze(1)

                Predicted_q_values = Predicted_q_vals_list[k].gather(1,actions[:,None].to(device)).squeeze(1)
                Y[dones] = 0
                Y = Y*self.discount_factor + rew.to(device)
                actual_loss = self.criteria(Predicted_q_values.double(),Y.double())
                propagated_loss = masks[:,k]*actual_loss
                loss = torch.sum(propagated_loss/total_used)
                count_losses.append(loss)

        loss = sum(count_losses)/self.n_heads
        loss.backward()

        for param in self.DQN.conv_net.parameters():
            if param.grad is not None:
                param.grad.data *= 1.0/self.n_heads

        nn.utils.clip_grad_norm_(self.DQN.parameters(),5)
        self.optimiser.step()


    def train(self):
        """
        Implement your training algorithm here
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        ep = self.ep
        for x in range(self.x,self.n_episodes):
            
            obs = self.next_obs
            done = self.done
            np.random.shuffle(self.head_list)
            terminal = self.terminal
            active_head = self.head_list[0]
            accumulated_rewards = 0
            while not terminal:
                
                if 0:
                    action = 1
                else:
                    action = self.make_action(obs,active_head,False)
                
                next_obs,reward,done,info = self.env.step(action)
                
                if info['ale.lives'] == 0:
                    terminal = True
                
                next_obs = np.transpose(next_obs,(2,0,1))
                masks = np.random.binomial(1,1,self.n_heads)
                accumulated_rewards+=reward
                
                self.push([obs,action,reward,done,next_obs,masks])
                
                self.current+=1
                self.current_train += 1
                self.current_target += 1
                obs = next_obs
                
                if self.current_train % self.train_buffer_size == 0 and len(self.buffer) > self.min_buffer_size:
                    self.learn()
                    self.current_train = 0

                if self.current_target > self.target_update_buffer and len(self.buffer) > self.min_buffer_size:
                    self.Target_DQN.load_state_dict(self.DQN.state_dict())
                    self.current_target = 0

                
                if self.current % self.Evaluation == 0:
                    print("\n","#" * 40, "Evaluation number %d"%(self.current/self.Evaluation),"#" * 40)
                    self.eval_num = self.current/self.Evaluation
                    env1 = Environment('BreakoutNoFrameskip-v4', self.args, atari_wrapper=True, test=True)
                    test(self,env1,total_episodes=100)
                    writer.add_scalar("Test/Max_test_Reward",self.max_test_reward,self.eval_num)
                    print("#" * 40, "Evaluation Ended!","#" * 40,"\n")
                
                if done:
                  self.reward_list.append(accumulated_rewards)
                  accumulated_rewards = 0
                  writer.add_scalar('Train/Episodic_Reward(Mean of last 30)',np.mean(self.reward_list[-30:]),ep+1)
                  ep+=1
            
            self.next_obs = np.transpose(self.env.reset(),(2,0,1))
            self.done = True
            self.terminal = False

            # writer.add_scalar('Train/Episodic_Reward(Mean of last 30)',np.mean(self.reward_list[-30:]),x+1)

            if len(self.reward_list) % 200 == 0:
                self.reward_list = self.reward_list[-150:]
            
            if (x+1)%20 == 0:
                print("Current = %d, episode = %d, Average_reward = %0.2f, epsilon = %0.2f"%(self.current, ep, np.mean(self.reward_list[-100:]), self.epsilon))
            
            if (x+1)%200 == 0:
                print("Saving_Weights_Model")
                torch.save({
                  'target_state_dict':self.Target_DQN.state_dict(),
                  'train_state_dict':self.DQN.state_dict(),
                  'optimiser_state_dict':self.optimiser.state_dict()
                  },Path_weights)
                
                print("Saving_Memory_Info")
                torch.save({
                  'current_info':[self.current,self.current_target,self.current_train],
                  'x':(x+1)*5,
                  'ep':ep+1,
                  'next_info':[self.next_obs,self.done,self.terminal],
                  'epsilon':self.epsilon,
                  # 'buffer':self.buffer,
                  #'reward_list':self.reward_list
                  }
                  ,Path_memory)






        
        ###########################