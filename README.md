# Deep Q-Network and its imporvements

*Please note, all code files should be placed in the same folder to run the program.*

How to reproduce our results?

All you have to do is just (For OS X and Linux Users):

```bash
mkdir results
pip install -r requirements.txt
python tuning.py
python exploration.py
python ablation.py
python improvements.py
```

If you use Windows, firstly create a file folder named `results`, and then open power shell:

```bash
pip install -r requirements.txt
python tuning.py
python exploration.py
python ablation.py
python improvements.py
```

If you just want to test whether our model is effective to solve `cart-pole v1` environment, or you want to change the parameters, create a new `.py` file and paste the following code:

```python
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

mean_reward, std_reward = run(env, config, n_repetitions=1)
Plot = LearningCurvePlot(title = 'Template')
Plot.add_curve(mean_reward, std_reward, label='test'))
Plot.save("results/test.png")

```

You can easily change the value of parameters in the dictionary object `baseline_config`. And after around 5 minutes, you can get a plot of the results.
