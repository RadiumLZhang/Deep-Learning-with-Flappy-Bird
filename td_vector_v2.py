# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 22:40:02 2019

@author: RadiumZhang
"""

#%%import important libraries
import os
import random
import gym
import sys
sys.path.append("Engine/gym-simpleflappy")
import time
import gym_simpleflappy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import pdb
#%%defining neural network model to get Q values for each action as an output 

class NeuralNetwork(nn.Module):

    def __init__(self):
        super(NeuralNetwork, self).__init__()
        #variables initialization
        self.number_of_actions = 2
        self.gamma = 0.99
        self.final_epsilon = 0.001 #changed from 0.0001 to 0.001 ; lalita
        self.initial_epsilon = 0.8 #changed from 0.3 to 0.8 by Lalita
        self.number_of_iterations = 2000000
        self.replay_memory_size = 10000
        self.minibatch_size = 64
        
        # Input Layer
        self.fc1 = nn.Linear(6, 20) #no. of in and out features
        self.relu1 = nn.ReLU(inplace=True)
        # Hidden Layer
        self.fc2 = nn.Linear(20,20)
        self.relu2 = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(20,20)
        self.relu3 = nn.ReLU(inplace=True)
        # Output Layer
        self.fc4 = nn.Linear(20, self.number_of_actions)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.relu3(out)
        out = self.fc4(out)
        return out
    
def init_weights(m):
#     if type(m) == nn.Conv2d or type(m) == nn.Linear:
    if type(m) == nn.Linear:
        torch.nn.init.uniform(m.weight, 0.0, 0.1)
        m.bias.data.fill_(0.01)
        
def tuple_to_tensor(tuple_1):

    tuple_tensor = np.array(tuple_1)
    tuple_tensor = tuple_tensor.astype(np.float32)
    tuple_tensor = torch.from_numpy(tuple_tensor)
#    if torch.cuda.is_available():  # put on GPU if CUDA is available
#        tuple_tensor = tuple_tensor.cuda()
    return tuple_tensor
#%%initialization of variables
C = 100
model = NeuralNetwork()
epsilon_decrements = np.linspace(model.initial_epsilon, model.final_epsilon, model.number_of_iterations)
#initialize replay memory
replay_memory = []
#initialize Q with random weights: i/p rndm weights in NN
model.apply(init_weights) #keep outside else learned weights will be init again
optimizer = optim.Adam(model.parameters(), lr=1e-6)
# initialize mean squared error loss
criterion = nn.MSELoss()
env = gym.make("SimpleFlappy-v0")
#%%building replay memory by taking random actions upto 10000(size of replay memory) iterations
#first loop is for episodes 1 to M
for i in range(model.number_of_iterations):
    state = env.reset()  #state = s1
    #init s1 in every episode
    state = tuple_to_tensor(state)
    score = 0
    while i<=model.replay_memory_size:
        
        env.render(mode='human') #uncommented by lalita
        
        #init Q value
        output = model(state)
        #init Q_hat
        output_hat = output
        action = torch.zeros([model.number_of_actions], dtype=torch.float32)
        #takes random action in init state
        action_index = torch.randint(model.number_of_actions, torch.Size([]), dtype=torch.int)
        action[action_index] = 1
        action_step = action_index
        state_1, reward, terminal, _ = env.step(action_step)
#        print(reward)
        
        score += reward
        state_1 = tuple_to_tensor(state_1)
        
        replay_memory.append((state,action,reward,state_1,terminal))
        state = state_1
        if terminal:
            state = env.reset()
            state = tuple_to_tensor(state)
#            
            print("iteration:", i,"elapsed time:", "Q max:", np.max(output.cpu().detach().numpy()),
                  'score = ', score)
#            score_plot.append(score)
            score = 0
        i += 1
#%%            
    #second loop runs until process is not terminated
    while terminal is False:
              
        #output for next state
        output = model(state_1)
        action = torch.zeros([model.number_of_actions], dtype=torch.float32)
                
        kk = 0#to use eps from index 0
        epsilon = epsilon_decrements[kk] 
        c = random.random()
        if (c <= epsilon):   
#                print("Performed random action!")
            action_index = torch.randint(model.number_of_actions, torch.Size([]), dtype=torch.int)
        else:
#                print("Performed greedy action")
            action_index = torch.argmax(output)
            
        kk += 1 #lalita
        action[action_index] = 1
      
        # Change action from vector to scalar, this is the action that we use
        action_step = action_index
        state_1, reward, terminal, _ = env.step(action_step)
        state_1 = tuple_to_tensor(state_1)
        score += reward
        reward = torch.tensor(reward).unsqueeze(0)
        #append (S_t,a_t,r_t,S_t+1) that is inside the while loop to replay memory
        
        replay_memory.append((state,action,reward,state_1,terminal))
        
        #if replay_memory is full
        if len(replay_memory) > model.replay_memory_size:
                replay_memory.pop(0) 
#        print(terminal)
        #for cal of y_j 
#        for i in range(number_of_state_1_batch):
        #sample (S_t,a_t,r_t,S_t+1) from minibatch
        
        
        #============ Copied From Previous Code ==============================
        if i % 10 == 0:
            minibatch = random.sample(replay_memory, min(len(replay_memory), model.minibatch_size))
#        print('memory:',replay_memory)
            # unpack minibatch
            state_batch = torch.cat(tuple(d[0] for d in minibatch))
            action_batch = torch.cat(tuple(d[1] for d in minibatch))
            reward_batch = torch.cat(tuple(d[2] for d in minibatch))
            state_1_batch = torch.cat(tuple(d[3] for d in minibatch))
            # get output for the next state
            number_of_state_1_batch = int(len(state_1_batch)/6)
            for num1 in range(number_of_state_1_batch):
                if num1 == 0:
                    output_1_batch = model(state_1_batch[6*num1:6*(num1+1)])
                else:
                    output_1_batch_i = model(state_1_batch[6*num1:6*(num1+1)])
                    output_1_batch = torch.cat((output_1_batch_i,output_1_batch),dim=0)
            output_1_batch = torch.reshape(output_1_batch,(number_of_state_1_batch,2)) 
   
            # set y_j to r_j for terminal state, otherwise to r_j + gamma*max(Q)
            y_batch = torch.cat(tuple(reward_batch[num1] if minibatch[num1][4]
                                      else reward_batch[num1] + model.gamma * torch.max(output_1_batch[num1])
                                      for num1 in range(len(minibatch))))
            
            
            # get output for the current state
            number_of_state_batch = int(len(state_batch)/6)
            for num in range(number_of_state_batch):
                if num == 0:
                    output_batch = model(state_batch[6*num:6*(num+1)])
                else:
                    output_batch_i = model(state_batch[6*num:6*(num+1)])
                    output_batch = torch.cat((output_batch_i,output_batch),dim=0)
            output_batch = torch.reshape(output_batch,(number_of_state_batch,2))       
    #        print('output_current',output_batch)
    #        print('action', action_batch)
            q_value = torch.sum(output_batch * action_batch, dim=1)
    
            # PyTorch accumulates gradients by default, so they need to be reset in each pass
            optimizer.zero_grad()
    
            # returns a new Tensor, detached from the current graph, the result will never require gradient
            y_batch = y_batch.detach()
    
            # calculate loss
            loss = criterion(q_value, y_batch)
    #        print('loss:',loss)
            # do backward pass
            loss.backward()
            optimizer.step()
        if i%C == 0:
            output_hat = model(state)
            
    if i % 25000 == 0:
        torch.save(model, "pretrained_model_tdvector2/current_model_" + str(i) + ".pth")
#
   
        

        
    
    
    


