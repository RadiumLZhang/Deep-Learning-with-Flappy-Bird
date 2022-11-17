#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 17:35:11 2019

@author: radiumzhang
"""


import os
import random
import gym
import sys
sys.path.append("Engine\gym-simpleflappy")
import time
import gym_simpleflappy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# import pdb
#%%=============================================================================

class NeuralNetwork(nn.Module):

    def __init__(self):
        super(NeuralNetwork, self).__init__()

        self.number_of_actions = 2
        self.gamma = 0.99
        self.final_epsilon = 0.0001
#        self.initial_epsilon = 0.1
        self.initial_epsilon = 0.1 #changed from 0.3 to 1 by Lalita
        self.number_of_iterations = 1000000
        self.replay_memory_size = 300000
#        self.minibatch_size = 32
        self.minibatch_size = 64
        
        # Input Layer
        self.conv1 = nn.Conv1d(4, 32, 2, stride = 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(32, 64, 2 , stride =1 )
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv1d(64, 64 ,2, stride =1 )
        self.relu3 = nn.ReLU(inplace=True)
        self.fc4 = nn.Linear(64, 16)
        self.relu4 = nn.ReLU(inplace=True)
        self.fc5 = nn.Linear(16, self.number_of_actions)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
#        print(out.shape)
        out = self.conv2(out)
        out = self.relu2(out)
#        print(out.shape)
        out = self.conv3(out)
        out = self.relu3(out)
#        print(out.shape)
        out = out.view(out.size()[0], -1)
        out = self.fc4(out)
        out = self.relu4(out)
#        print(out.shape)
        out = self.fc5(out)
#        print(out.shape)

        return out


def init_weights(m):
#     if type(m) == nn.Conv2d or type(m) == nn.Linear:
    if type(m) == nn.Linear or type(m) == nn.Conv1d:
#        torch.nn.init.uniform(m.weight, -0.01, 0.01)
        torch.nn.init.uniform(m.weight, -0.01, 0.01)
        m.bias.data.fill_(0.01)

def tuple_to_tensor(tuple_1):
#     image_tensor = image.transpose(2, 0, 1)
#     tuple_tensor = tuple_tensor.astype(np.float32)
    #tuple_tensor = 
    tuple_tensor = np.array(tuple_1)
#    tuple_tensor = tuple_tensor.astype(np.float32)
    tuple_tensor = tuple_tensor.astype(np.float32)
    tuple_tensor = torch.from_numpy(tuple_tensor)
    if torch.cuda.is_available():  # put on GPU if CUDA is available
        tuple_tensor = tuple_tensor.cuda()
    return tuple_tensor
#%%=============================================================================
#Training Function
def train(model, start):
    # define Adam optimizer
    score = 0
#    score_plot = []
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
    #state = torch.cat((state,state)).unsqueeze(0)
#    print(state.shape)
#    print(state.shape)
    #state = torch.cat(state).unsqueeze(0)

    
    # initialize replay memory
    replay_memory = []

#    # initial action is do nothing
#    action = torch.zeros([model.number_of_actions], dtype=torch.float32)
#    action[0] = 1
#    image_data, reward, terminal = game_state.frame_step(action)
#    image_data = resize_and_bgr2gray(image_data)
#    image_data = image_to_tensor(image_data)
#    state = torch.cat((image_data, image_data, image_data, image_data)).unsqueeze(0)

    # initialize epsilon value
    epsilon = model.initial_epsilon
#    beta1 = 0.05
#    beta2 = 0.2
#    beta3 = 0.4
    iteration = 0
    
    epsilon_decrements = np.linspace(model.initial_epsilon, model.final_epsilon, model.number_of_iterations)
    # main infinite loop
    while iteration < model.number_of_iterations:
       # Epsilon annealing scheme: 
#        if iteration > 10000 and iteration <= 100000:
#            epsilon  = iteration**(-beta1)
#        elif iteration > 100000 and iteration <= 1000000:
#            epsilon = iteration**(-beta2)
#        elif iteration > 1000000:
#            epsilon = iteration**(-beta3)
        
        # env.render(mode='human') #uncommented by lalita
        # get output from the neural network
        output = model(state)
#        print(output)

        action = torch.zeros([model.number_of_actions], dtype=torch.float32)
#        print('initial_actin:',action)
        if torch.cuda.is_available():  # put on GPU if CUDA is available
            action = action.cuda()     
            
#        if iteration < model.replay_memory_size/10:
#            action_index = torch.randint(model.number_of_actions, torch.Size([]), dtype=torch.int)
#            if torch.cuda.is_available():  # put on GPU if CUDA is available
#                action_index = action_index.cuda()
#            action[action_index] = 1
#            action_step = action_index
##            print(action_index)
#            state_1, reward, terminal, _ = env.step(action_step)
#            if reward == 4.0:
#                score += 1
#            state_1 = tuple_to_tensor(state_1)   
#            action = action.unsqueeze(0)
##            print(action)
#            reward = torch.from_numpy(np.array([reward], dtype=np.float32)).unsqueeze(0)
#            
#        # save transition to replay memory
#            replay_memory.append((state, action, reward, state_1, terminal))
#
#        # if replay memory is full, remove the oldest transition
#            if len(replay_memory) > model.replay_memory_size:
#                replay_memory.pop(0)
##            epsilon = epsilon_decrements[iteration]
#            
#        else:
#        if iteration % 1 == 0:
        c = random.random()
        if (c <= epsilon):   
#                    print("Performed random action!")
            action_index = torch.randint(model.number_of_actions, torch.Size([]), dtype=torch.int)
#            action_randomed = 1

        else:
#                    print("greedy action")
            action_index = torch.argmax(output)
#            action_randomed = 0
                
        if torch.cuda.is_available():  # put on GPU if CUDA is available
            action_index = action_index.cuda()
#        else:
#            action_index = 0
#            print(action_index)
        action[action_index] = 1
        # Change action from vector to scalar, this is the action that we use
        action_step = action_index
#        print('step_action:',action_step)
        # get next state and reward
        state_1, reward, terminal, _ = env.step(action_step)
        state_1_int = tuple_to_tensor(state_1)
#        state_1_int = torch.reshape(state_1_int,(1,6)) 
#        print('state', state)
#        print(state_1_int)
#        print(state_1_int.shape)
#        print(state.squeeze(0)[1,:])
#        print(state.squeeze(0)[1,:].shape)
#        print(state.shape)
        state_1 = torch.stack((state.squeeze(0)[1,:],state.squeeze(0)[2,:],state.squeeze(0)[3,:], state_1_int)).unsqueeze(0)
#        print('state_1:',state_1)
#        print(state_1.shape)
        if reward == 4.0:
            score  += 1
        action = action.unsqueeze(0)
        #assert reward==-10, "Bird about to die"
        # CHECKS for Reward
        # pdb.set_trace()
        
        reward = torch.from_numpy(np.array([reward], dtype=np.float32)).unsqueeze(0)
        # save transition to replay memory
        replay_memory.append((state, action, reward, state_1, terminal))
        
        # if replay memory is full, remove the oldest transition
        if len(replay_memory) > model.replay_memory_size:
            replay_memory.pop(0)

        # epsilon annealing
        epsilon = epsilon_decrements[iteration]
#        epsilon  = max(model.final_epsilon, (iteration+1)**(-0.3))
        # sample random minibatch
        
        
        if iteration % 1 ==0:
            minibatch = random.sample(replay_memory, min(len(replay_memory), model.minibatch_size))
    #        print('memory:',replay_memory)
            # unpack minibatch
    #        state_batch = torch.stack(tuple(d[0] for d in minibatch),0)
            state_batch = torch.cat(tuple(d[0] for d in minibatch))
    #        print(len(state_batch))
            action_batch = torch.cat(tuple(d[1] for d in minibatch))
            reward_batch = torch.cat(tuple(d[2] for d in minibatch))
            state_1_batch = torch.cat(tuple(d[3] for d in minibatch))
        
    #         print('action',action_batch)
    #        print('state',state_1_batch)
            if torch.cuda.is_available():  # put on GPU if CUDA is available
                state_batch = state_batch.cuda()
                action_batch = action_batch.cuda()
                reward_batch = reward_batch.cuda()
                state_1_batch = state_1_batch.cuda()
        
            # get output for the next state
            output_1_batch = model(state_1_batch)
    #        number_of_state_1_batch = int(len(state_1_batch)/6)
    #        for i in range(number_of_state_1_batch):
    #            if i == 0:
    #                output_1_batch = model(state_1_batch[6*i:6*(i+1)])
    #            else:
    #                output_1_batch_i = model(state_1_batch[6*i:6*(i+1)])
    ##                print('next_state:',state_1_batch[6*i:6*(i+1)])
    #                output_1_batch = torch.cat((output_1_batch,output_1_batch_i),dim=0) #change the position of two
    ##                print('output11batch for',i,' th', output_1_batch_i)
    ##                print('current output1 batch',output_1_batch)
    #        output_1_batch = torch.reshape(output_1_batch,(number_of_state_1_batch,2)) 
             
    #        print('output_1',output_1_batch)
    #        for i in range(number_of_state_1_batch):
    #            print('cuurent max output is:',torch.max(output_1_batch[i]))
            # set y_j to r_j for terminal state, otherwise to r_j + gamma*max(Q)
            y_batch = torch.cat(tuple(reward_batch[i] if minibatch[i][4]
                                      else reward_batch[i] + model.gamma * torch.max(output_1_batch[i])
                                      for i in range(len(minibatch))))
            
            
    #        print('y_batch',y_batch)
            # extract Q-value
            # get output for the current state
    #        number_of_state_batch = int(len(state_batch)/6)
    #        print(number_of_state_batch)
    #        for i in range(number_of_state_batch):
    #            if i == 0:
    #                output_batch = model(state_batch[6*i:6*(i+1)])
    #            else:
    #                output_batch_i = model(state_batch[6*i:6*(i+1)])
    #                output_batch = torch.cat((output_batch,output_batch_i),dim=0)
    ##                print('outputbatch for',i,' th', output_batch_i)
    ##        print('output_batch before reshape',output_batch)
    #        output_batch = torch.reshape(output_batch,(number_of_state_batch,2))       
    #        print('output_current',output_batch)
    #        print('action', action_batch)
    #        print('y*a:',output_batch * action_batch)
            q_value = torch.sum(model(state_batch) * action_batch, dim=1)
    #        print('q_value',q_value)
            # PyTorch accumulates gradients by default, so they need to be reset in each pass
            optimizer.zero_grad()
    
            # returns a new Tensor, detached from the current graph, the result will never require gradient
            y_batch = y_batch.detach()
    #        print('y_batch after detach',y_batch)
            # calculate loss
            loss = criterion(q_value, y_batch)
    #        print('loss:',loss)
            # do backward pass
            loss.backward()
            optimizer.step()
#        print('model parameter:',model.parameters())
        # set state to be state_1
        state = state_1
            #print(state,terminal)
        iteration += 1
        Q_max = np.max(output.cpu().detach().numpy())
        Q_max_list.append(Q_max)
        score_list.append(score)
        if terminal:
            state = env.reset()
            state = tuple_to_tensor(state)
            state = torch.stack((state_int, state_int, state_int, state_int)).unsqueeze(0)
    #            print("iteration:", iteration, "elapsed time:", time.time() - start, "epsilon:", epsilon, "action:",
    #              action_index.cpu().detach().numpy(), "reward:", reward.numpy()[0][0], "Q max:",
    #              np.max(output.cpu().detach().numpy()),'score = ', score)
            print("iteration:", iteration,"epsilon", epsilon, "elapsed time:", time.time() - start, "Q max:", Q_max, 'loss', loss,
                  'score = ', score)
#                score_plot.append(score)
            score = 0
        
        if iteration % 25000 == 0:
            torch.save(model, "pretrained_model_vector_new/current_model_" + str(iteration) + ".pth")
    #
#        print("iteration:", iteration, "elapsed time:", time.time() - start, "epsilon:", epsilon, "action:",
#              action_index.cpu().detach().numpy(), "Random action:",action_randomed,"reward:", reward.numpy()[0][0], "Q max:",
#              np.max(output.cpu().detach().numpy()),'score = ', score)
    
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
#     game_state = GameState()

#     # initial action is do nothing
#     action = torch.zeros([model.number_of_actions], dtype=torch.float32)
#     action[0] = 1
#     image_data, reward, terminal = game_state.frame_step(action)
#     image_data = resize_and_bgr2gray(image_data)
#     image_data = image_to_tensor(image_data)
#     state = torch.cat((image_data, image_data, image_data, image_data)).unsqueeze(0)

    while True:
        # get output from the neural network
        env.render(mode='human')
        output = model(state)
        print('MODEL:',model)
        action = torch.zeros([model.number_of_actions], dtype=torch.float32)
        if torch.cuda.is_available():  # put on GPU if CUDA is available
            action = action.cuda()

        # get action
        action_index = torch.argmax(output)
        print('output:',output)
        print('action_index:', action_index)
        if torch.cuda.is_available():  # put on GPU if CUDA is available
            action_index = action_index.cuda()
        action[action_index] = 1
        print('action:',action)
        action_step = action_index
        state_1, reward, terminal, _ = env.step(action_step)
        state_1 = tuple_to_tensor(state_1)
        state_1 = torch.stack((state_int, state_1, state_1, state_1)).unsqueeze(0)
#         # get next state
#         image_data_1, reward, terminal = game_state.frame_step(action)
#         image_data_1 = resize_and_bgr2gray(image_data_1)
#         image_data_1 = image_to_tensor(image_data_1)
#         state_1 = torch.cat((state.squeeze(0)[1:, :, :], image_data_1)).unsqueeze(0)

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
            'pretrained_model_vector_new/current_model_1000000.pth',
            map_location='cpu' if not cuda_is_available else None
        ).eval()

        if cuda_is_available:  # put on GPU if CUDA is available
            model = model.cuda()

        test(model)

    elif mode == 'train':
        if not os.path.exists('pretrained_model_vector_new/'):
            os.mkdir('pretrained_model_vector_new/')

        model = NeuralNetwork()
#        model = nn.RNN(32,20,2)

        if cuda_is_available:  # put on GPU if CUDA is available
            model = model.cuda()
        
#        print("init_weights",init_weights)
        model.apply(init_weights)
#        print(model)
        start = time.time()

        [Q,S] = train(model, start)
        return Q, S
##
        
#if __name__ == "__main__":
#    main(sys.argv[1])
#%%
#main('test')      
# [Q,S] = main('train')

# #%%
# import matplotlib.pyplot as plt
# plt.figure(1)
# plt.plot(S)
# plt.figure(2)
# plt.plot(Q)


#%%
main('test')