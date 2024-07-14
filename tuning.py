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
                   "alpha": 0.01,                                 ## learning rate of Network
                   "gamma": 0.99,                                 ## discount rate of Network
                   "factor": 2,                                   ## exploration rate for different exploration policy
                   "capacity": 10000,                             ## capacity of memory pool
                   "batch_size": 64,                              ## batch size for learning
                   "device": "cpu",                               ## if you want to use GPU, on Apple Silicon, set it as "mps", else set it as "cuda"
                   "replay": True,                                ## whether to use experience replay
                   "target": True,                                ## whether to use target network
                   "exploration": "softmax",                      ## must be "ucb" or "e-greedy" or "softmax"
                   "layers": 1,                                   ## number of layers of q-network
                   "neurons": 128,                                ## number of neurons of network, must be int or list with length layers+1
                   "activation": nn.ReLU(),                       ## activation method
                   "initialization": nn.init.xavier_normal_,      ## initialization method
                   "if_conv": False,                              ## whether to use convolutional layers, only effective for Dueling DQN
                   "update_policy": "soft",                       ## how to update target network, hard or soft
                   "update_para": 0.01,                           ## update parameter for target network updating
                   "annealing": False}                            ## whether to use annealing method, only effective for "softmax" and "e-greedy" policy


## Tune Architecture of DQN

with open("results/architecture.txt", "w") as f:
    f.write("Mean rewards and their standard error of different network architecture.\n")

def tuneArich(layer, neuron):
    env = gym.make("CartPole-v1")
    config = deepcopy(baseline_config)
    config['layers'] = layer
    config['neurons'] = neuron
    mean_reward, std_reward = run(env, config)
    with open("results/architecture.txt", "a") as f:
        f.write("Layer = {}, Neuron = {}, Mean Reward = {:.3f}, Std Error = {:.3f} \n".format(layer, neuron, mean_reward[-1], std_reward[-1]))

layers = [1, 2, 3, 4, 5]
neurons = [16, 32, 64, 128, 256]

threads = []
for layer in layers:
    for neuron in neurons:
        tuneArich(layer, neuron)


## Tune Initialization Function
Plot = LearningCurvePlot(title = 'DQNs with Different Initialization Methods')
PlotNoError = LearningCurvePlotNoError(title = 'DQNs with Different Initialization Methods')
initializations = [nn.init.uniform_,
                   nn.init.normal_,
                   nn.init.zeros_,
                   nn.init.ones_,
                   nn.init.xavier_normal_,
                   nn.init.xavier_uniform_,
                   nn.init.kaiming_normal_,
                   nn.init.kaiming_uniform_]
initializations_names = ['uniform', 'normal', 'zeros', 'ones',
                         'xavier normal', 'xavier uniform', 'kaiming normal', 'kaiming uniform']
for act in range(len(initializations)):
    config = deepcopy(baseline_config) # we must use deepcope to avoid changing the value of original baseline config
    config['initialization'] = initializations[act]
    mean_reward, std_reward = run(env, config)
    Plot.add_curve(mean_reward, std_reward, label='{}'.format(initializations_names[act]))
    PlotNoError.add_curve(mean_reward, std_reward, label='{}'.format(initializations_names[act]))

Plot.save("results/Initializations.png")
PlotNoError.save("results/InitializationsNoError.png")

## Tune Activation Function
Plot = LearningCurvePlot(title = 'DQNs with Different Activation Functions')
PlotNoError = LearningCurvePlotNoError(title = 'DQNs with Different Activation Functions')

activations = [nn.ReLU(), nn.Sigmoid(), nn.Tanh(), nn.SiLU()]
activation_names = ['ReLU', 'Sigmoid', 'Tanh', 'SiLU']

for act in range(len(activations)):
    config = deepcopy(baseline_config) # we must use deepcope to avoid changing the value of original baseline config
    config['activation'] = activations[act]
    mean_reward, std_reward = run(env, config)
    Plot.add_curve(mean_reward, std_reward, label=r'{}'.format(activation_names[act]))
    PlotNoError.add_curve(mean_reward, std_reward, label=r'{}'.format(activation_names[act]))
    Plot.save("results/Activations.png")
    PlotNoError.save("results/ActivationsNoError.png")

Plot.save("results/Activations.png")
PlotNoError.save("results/ActivationsNoError.png")


## Tune Learning Rate
Plot = LearningCurvePlot(title = 'DQNs with Different Learning Rates')
PlotNoError = LearningCurvePlotNoError(title = 'DQNs with Different Learning Rates')
lrs = [1, 0.1, 0.01, 0.001, 0.0001]
for act in range(len(lrs)):
    config = deepcopy(baseline_config) # we must use deepcope to avoid changing the value of original baseline config
    config['alpha'] = lrs[act]
    mean_reward, std_reward = run(env, config)
    Plot.add_curve(mean_reward, std_reward, label=r'$\alpha$ = {}'.format(lrs[act]))
    PlotNoError.add_curve(mean_reward, std_reward, label=r'$\alpha$ = {}'.format(lrs[act]))
    Plot.save("results/LearningRates.png")
    PlotNoError.save("results/LearningRatesNoError.png")

Plot.save("results/LearningRates.png")
PlotNoError.save("results/LearningRatesNoError.png")

## Tune Batch Size
Plot = LearningCurvePlot(title = 'DQNs with Different Batch Size Values')
PlotNoError = LearningCurvePlotNoError(title = 'DQNs with Different Batch Size Values')
batch_sizes = [8, 16, 64, 128, 256, 1024]
for act in range(len(batch_sizes)):
    config = deepcopy(baseline_config) # we must use deepcope to avoid changing the value of original baseline config
    config['batch_size'] = batch_sizes[act]
    mean_reward, std_reward = run(env, config)
    Plot.add_curve(mean_reward, std_reward, label=r'$B$ = {}'.format(batch_sizes[act]))
    PlotNoError.add_curve(mean_reward, std_reward, label=r'$B$ = {}'.format(batch_sizes[act]))
    Plot.save("results/BatchSize.png")
    PlotNoError.save("results/BatchSizeNoError.png")

Plot.save("results/BatchSize.png")
PlotNoError.save("results/BatchSizeNoError.png")


## Tune Capacity Size
Plot = LearningCurvePlot(title = 'DQNs with Different Size of Capacity')
PlotNoError = LearningCurvePlotNoError(title = 'DQNs with Different Size of Capacity')
V = [100, 1000, 10000, 100000, 1000000]
for act in range(len(V)):
    config = deepcopy(baseline_config) # we must use deepcope to avoid changing the value of original baseline config
    config['capacity'] = V[act]
    mean_reward, std_reward = run(env, config)
    Plot.add_curve(mean_reward, std_reward, label=r'$V$ = {}'.format(V[act]))
    PlotNoError.add_curve(mean_reward, std_reward, label=r'$V$ = {}'.format(V[act]))
    Plot.save("results/Capacity.png")
    PlotNoError.save("results/CapacityNoError.png")

Plot.save("results/Capacity.png")
PlotNoError.save("results/CapacityNoError.png")

## Tune Discount Rate
Plot = LearningCurvePlot(title = 'DQNs with Different Discount Rates')
PlotNoError = LearningCurvePlotNoError(title = 'DQNs with Different Discount Rates')
discountRates = [0.1, 0.5, 0.8, 0.85, 0.9, 0.95, 0.99, 1]
for act in range(len(discountRates)):
    config = deepcopy(baseline_config) # we must use deepcope to avoid changing the value of original baseline config
    config['gamma'] = discountRates[act]
    mean_reward, std_reward = run(env, config)
    Plot.add_curve(mean_reward, std_reward, label=r'$\gamma$ = {}'.format(discountRates[act]))
    PlotNoError.add_curve(mean_reward, std_reward, label=r'$\gamma$ = {}'.format(discountRates[act]))
    Plot.save("results/DiscountRates.png")
    PlotNoError.save("results/DiscountRatesNoError.png")

Plot.save("results/DiscountRates.png")
PlotNoError.save("results/DiscountRatesNoError.png")

## Tune Update Parameters of Hard Updating
Plot = LearningCurvePlot(title = 'DQNs with Different Update Parameters (Hard Update)')
PlotNoError = LearningCurvePlotNoError(title = 'DQNs with Different Update Parameters (Hard Update)')
baseline_config['update_policy'] = 'hard'
hard_paras = [1, 10, 100, 1000, 10000, 100000]
for act in range(len(hard_paras)):
    config = deepcopy(baseline_config) # we must use deepcope to avoid changing the value of original baseline config
    config['update_para'] = hard_paras[act]
    mean_reward, std_reward = run(env, config)
    Plot.add_curve(mean_reward, std_reward, label=r'$t$ = {}'.format(hard_paras[act]))
    PlotNoError.add_curve(mean_reward, std_reward, label=r'$t$ = {}'.format(hard_paras[act]))
    Plot.save("results/HardParas.png")
    PlotNoError.save("results/HardParasNoError.png")

Plot.save("results/HardParas.png")
PlotNoError.save("results/HardParasNoError.png")


## Tune Update Parameters of Soft Updating
Plot = LearningCurvePlot(title = 'DQNs with Different Update Parameters (Soft Update)')
PlotNoError = LearningCurvePlotNoError(title = 'DQNs with Different Update Parameters (Soft Update)')
soft_paras = [1, 0.1, 0.01, 0.001, 0.0001, 0.00001]
for act in range(len(soft_paras)):
    config = deepcopy(baseline_config) # we must use deepcope to avoid changing the value of original baseline config
    config['update_para'] = soft_paras[act]
    mean_reward, std_reward = run(env, config)
    Plot.add_curve(mean_reward, std_reward, label=r'$\tau$ = {}'.format(soft_paras[act]))
    PlotNoError.add_curve(mean_reward, std_reward, label=r'$\tau$ = {}'.format(soft_paras[act]))
    Plot.save("results/SoftParas.png")
    PlotNoError.save("results/SoftParasNoError.png")

Plot.save("results/SoftParas.png")
PlotNoError.save("results/SoftParasNoError.png")




