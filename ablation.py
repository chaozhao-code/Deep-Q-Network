from model import *
from util import LearningCurvePlot, LearningCurvePlotNoError
import random
import numpy as np
import torch
from torch import nn
import os
import gymnasium as gym
from copy import deepcopy

## Set Random Seed
env = gym.make("CartPole-v1")
SEED = 22032022
os.environ["PYTHONHASHSEED"] = str(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
env.action_space.seed(SEED)

baseline_config = {"type": "DQN",                                 ## must be "DQN", "DDQN" or "DuelingDQN"
                   "alpha": 0.001,                                 ## learning rate of Network
                   "gamma": 1,                                 ## discount rate of Network
                   "factor": 2,                                   ## exploration rate for different exploration policy
                   "capacity": 1000000,                             ## capacity of memory pool
                   "batch_size": 256,                              ## batch size for learning
                   "device": "cpu",                               ## if you want to use GPU, on Apple Silicon, set it as "mps", else set it as "cuda"
                   "replay": True,                                ## whether to use experience replay
                   "target": True,                                ## whether to use target network
                   "exploration": "ucb",                      ## must be "ucb" or "e-greedy" or "softmax"
                   "layers": 1,                                   ## number of layers of q-network
                   "neurons": 16,                                ## number of neurons of network, must be int or list with length layers+1
                   "activation": nn.SiLU(),                       ## activation method
                   "initialization": nn.init.uniform_,      ## initialization method
                   "if_conv": False,                              ## whether to use convolutional layers, only effective for Dueling DQN
                   "update_policy": "soft",                       ## how to update target network, hard or soft
                   "update_para": 0.1,                           ## update parameter for target network updating
                   "annealing": False}                            ## whether to use annealing method, only effective for "softmax" and "e-greedy" policy



## Tune Activation Function
Plot = LearningCurvePlot(title = r'Ablation Study')
PlotNoError = LearningCurvePlotNoError(title = r'Ablation Study')


## DQN
config = deepcopy(baseline_config)
config['replay'] = False
config['target'] = False
mean_reward, std_reward = run(env, config)
Plot.add_curve(mean_reward, std_reward, label=r'DQN')
PlotNoError.add_curve(mean_reward, std_reward, label=r'DQN')



## DQN-ER
config = deepcopy(baseline_config)
config['replay'] = True
config['target'] = False
mean_reward, std_reward = run(env, config)
Plot.add_curve(mean_reward, std_reward, label=r'DQN-ER')
PlotNoError.add_curve(mean_reward, std_reward, label=r'DQN-ER')


## DQN-TN
config = deepcopy(baseline_config)
config['replay'] = False
config['target'] = True
mean_reward, std_reward = run(env, config)
Plot.add_curve(mean_reward, std_reward, label=r'DQN-TN')
PlotNoError.add_curve(mean_reward, std_reward, label=r'DQN-TN')


## DQN-TN-ER
config = deepcopy(baseline_config)
config['replay'] = True
config['target'] = True
mean_reward, std_reward = run(env, config)
Plot.add_curve(mean_reward, std_reward, label=r'DQN-ER-TN')
PlotNoError.add_curve(mean_reward, std_reward, label=r'DQN-ER-TN')


Plot.save("results\AblationStudy.png")
PlotNoError.save("results\AblationStudyNoError.png")





