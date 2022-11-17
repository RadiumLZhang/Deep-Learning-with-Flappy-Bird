# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 17:31:56 2019



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 17:35:11 2019



import os
import random
import gym
import sys
sys.path.append("engine/gym-simpleflappy-master")
import time
import gym_simpleflappy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import pdb
#%%=============================================================================

class NeuralNetwork(nn.Module):

    def __init__(self):
        super(NeuralNetwork, self).__init__()

        self.number_of_actions = 2
        self.gamma = 0.99
        self.final_epsilon = 0.0001

        self.initial_epsilon = 0.1 
        self.number_of_iterations = 2000000
        self.replay_memory_size = 1000000
        self.minibatch_size = 32
        
        self.conv1 = nn.Conv1d(4, 16, 2, stride = 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(16, 16, 3 , stride =1 )
        self.relu2 = nn.ReLU(inplace=True)

        self.fc4 = nn.Linear(16, 16)
        self.relu4 = nn.ReLU(inplace=True)
        self.fc5 = nn.Linear(16, self.number_of_actions)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)

        out = out.view(out.size()[0], -1)
        out = self.fc4(out)
        out = self.relu4(out)

        out = self.fc5(out)


        return out


def init_weights(m):

    if type(m) == nn.Linear or type(m) == nn.Conv1d:
        torch.nn.init.uniform(m.weight, -0.01, 0.01)
        m.bias.data.fill_(0.01)

def tuple_to_tensor(tuple_1):
    tuple_tensor = np.array(tuple_1)
    tuple_tensor = tuple_tensor.astype(np.float32)
    tuple_tensor = torch.from_numpy(tuple_tensor)
    if torch.cuda.is_available():  
        tuple_tensor = tuple_tensor.cuda()
    return tuple_tensor
#%%=============================================================================
#Training Function
def train(model, start):
    # define Adam optimizer
    score = 0
    optimizer = optim.Adam(model.parameters(), lr=1e-6)

    # initialize mean squared error loss
    criterion = nn.MSELoss()
    Q_max_list = []
    score_list = []
    env = gym.make("SimpleFlappy-v0")
    # instantiate game
    state = env.reset()
    state_int = tuple_to_tensor(state)
    state = torch.stack((state_int, state_int, state_int, state_int)).unsqueeze(0)

    
    # initialize replay memory
    replay_memory = []

#    # initial action is do nothing
    # initialize epsilon value
    epsilon = model.initial_epsilon
    iteration = 0
    
    epsilon_decrements = np.linspace(model.initial_epsilon, model.final_epsilon, model.number_of_iterations)
    # main infinite loop
    while iteration < model.number_of_iterations:
         
       # get output from the neural network
        output = model(state)
        action = torch.zeros([model.number_of_actions], dtype=torch.float32)
        if torch.cuda.is_available():  
            action = action.cuda()     

        c = random.random()
        if (c <= epsilon):   
                 
            action_index = torch.randint(model.number_of_actions, torch.Size([]), dtype=torch.int)

        else:
            action_index = torch.argmax(output)

                
        if torch.cuda.is_available():  
            action_index = action_index.cuda()

        action[action_index] = 1
       
        action_step = action_index
        state_1, reward, terminal, _ = env.step(action_step)
        state_1_int = tuple_to_tensor(state_1)
        state_1 = torch.stack((state.squeeze(0)[1,:],state.squeeze(0)[2,:],state.squeeze(0)[3,:], state_1_int)).unsqueeze(0)

        if reward == 4.0:
            score  += 1
        action = action.unsqueeze(0)
      
        reward = torch.from_numpy(np.array([reward], dtype=np.float32)).unsqueeze(0)
        # save transition to replay memory
        replay_memory.append((state, action, reward, state_1, terminal))
        
        # if replay memory is full, remove the oldest transition
        if len(replay_memory) > model.replay_memory_size:
            replay_memory.pop(0)

        # epsilon annealing
        epsilon = epsilon_decrements[iteration]

        # sample random minibatch
        
        
        if iteration % 1 ==0:
            minibatch = random.sample(replay_memory, min(len(replay_memory), model.minibatch_size))
            state_batch = torch.cat(tuple(d[0] for d in minibatch))
            action_batch = torch.cat(tuple(d[1] for d in minibatch))
            reward_batch = torch.cat(tuple(d[2] for d in minibatch))
            state_1_batch = torch.cat(tuple(d[3] for d in minibatch))
        
            if torch.cuda.is_available(): 
                state_batch = state_batch.cuda()
                action_batch = action_batch.cuda()
                reward_batch = reward_batch.cuda()
                state_1_batch = state_1_batch.cuda()
        
            # get output for the next state
            output_1_batch = model(state_1_batch)
  
            y_batch = torch.cat(tuple(reward_batch[i] if minibatch[i][4]
                                      else reward_batch[i] + model.gamma * torch.max(output_1_batch[i])
                                      for i in range(len(minibatch))))
            
            
            q_value = torch.sum(model(state_batch) * action_batch, dim=1)

            # PyTorch accumulates gradients by default, so they need to be reset in each pass
            optimizer.zero_grad()
    
            # returns a new Tensor, detached from the current graph, the result will never require gradient
            y_batch = y_batch.detach()
    
            # calculate loss
            loss = criterion(q_value, y_batch)

            # do backward pass
            loss.backward()
            optimizer.step()

        # set state to be state_1
        state = state_1
            
        iteration += 1
        Q_max = np.max(output.cpu().detach().numpy())
        Q_max_list.append(Q_max)
        score_list.append(score)
        if terminal:
            state = env.reset()
            state_int = tuple_to_tensor(state)
            state = torch.stack((state_int, state_int, state_int, state_int)).unsqueeze(0)
    
            print("iteration:", iteration,"epsilon", epsilon, "elapsed time:", time.time() - start, "Q max:", Q_max, 'loss', loss,
                  'score = ', score)
            score = 0
        
        if iteration % 25000 == 0:
            torch.save(model, "pretrained_model_vector_midnight/current_model_" + str(iteration) + ".pth")
    
    env.close()
    return Q_max_list, score_list
#%%=============================================================================
#Test Function

def test(model):
    env = gym.make("SimpleFlappy-v0")
    # instantiate game
    state = env.reset()
    state_int = tuple_to_tensor(state)
    state = torch.stack((state_int, state_int, state_int, state_int)).unsqueeze(0)

    score = 0
    start = time.time()
    while True:
        # get output from the neural network

        env.render(mode='human')
        output = model(state)

        action = torch.zeros([model.number_of_actions], dtype=torch.float32)
        if torch.cuda.is_available():  
            action = action.cuda()

        # get action
        action_index = torch.argmax(output)

        if torch.cuda.is_available(): 
            action_index = action_index.cuda()
        action[action_index] = 1

        action_step = action_index
        state_1, reward, terminal, _ = env.step(action_step)
        state_1_int = tuple_to_tensor(state_1)
        state_1 = torch.stack((state.squeeze(0)[1,:],state.squeeze(0)[2,:],state.squeeze(0)[3,:], state_1_int)).unsqueeze(0)
        if reward == 4.0:
            score += 1
            print('### Score = ',score, '### Time elapsed = ', time.time() - start, '###')
#         # get next state


        # set state to be state_1
        state = state_1
        if terminal:
            state = env.reset()
            state_int = tuple_to_tensor(state)
            state = torch.stack((state_int, state_int, state_int, state_int)).unsqueeze(0)
            
    env.close()    
#%%=============================================================================
def main(mode):
    cuda_is_available = torch.cuda.is_available()

    if mode == 'test':
        model = torch.load(
            'pretrained_model_vector_midnight/current_model_2000000.pth',
            map_location='cpu' if not cuda_is_available else None
        ).eval()

        if cuda_is_available:  
            model = model.cuda()

        test(model)

    elif mode == 'train':
        if not os.path.exists('pretrained_model_vector/'):
            os.mkdir('pretrained_model_vector/')

        model = NeuralNetwork()


        if cuda_is_available: 
            model = model.cuda()
        

        model.apply(init_weights)

        start = time.time()

        [Q,S] = train(model, start)
        return Q, S

#%%
[Q,S] = main('train')
main('test')      


#%%
import matplotlib.pyplot as plt
plt.figure(1)
plt.plot(S)
plt.figure(2)
plt.plot(Q)


    
