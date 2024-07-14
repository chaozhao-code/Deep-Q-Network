import random
import sys

import numpy as np
import torch

from torch import nn
from collections import deque


from util import softmax, linear_anneal, smooth
import tqdm

class DQN(nn.Module):
  def __init__(self, n_states, n_actions, layers=1, neurons=128, activation=nn.ReLU(), initialization=nn.init.zeros_):
    '''
    Initialization of DQN
    :param n_states: the dimension of state space
    :param n_actions: the dimension of action space
    :param layers: besides the input and output layer, the number of hidden layers
    :param neurons: if its type is int, which means the number of neurons of hidden layers,
                    if its type is list, which means the number of neurons in each layer
    :param activation: torch type, activation functions
    '''
    super(DQN, self).__init__()
    self.initialization = initialization
    modules = []
    if type(neurons) == int:
      modules.append(nn.Linear(n_states, neurons))
      modules.append(activation)
      for i in range(layers):
        modules.append(nn.Linear(neurons, neurons))
        modules.append(activation)
      modules.append(nn.Linear(neurons, n_actions))
    elif type(neurons) == list:
      if len(neurons) != 1 + layers:
        raise KeyError("Length of neurons must be (layers+1)")
      modules.append(nn.Linear(n_states, neurons[0]))
      modules.append(activation)
      j = 0
      for i in range(layers):
        modules.append(nn.Linear(neurons[j], neurons[j+1]))
        modules.append(activation)
        j = j + 1
      modules.append(nn.Linear(neurons[-1], n_actions))
    else:
      raise TypeError("Only Int and List Are Allowed")
    self.network = nn.Sequential(*modules)
    self.network.apply(self.initialize_weights)

  def initialize_weights(self, module):
    if isinstance(module, nn.Linear):
      self.initialization(module.weight)
      ## always initialize bias as 0
      nn.init.zeros_(module.bias)

  def forward(self, state):
    """
    calculate the Q-value of given state s
    :param state: s, size: n_states
    :return: Q-value, size: n_actions
    """
    return self.network(state)

class DuelingDQN(nn.Module):
  def __init__(self, n_states, n_actions, initialization=nn.init.zeros_, if_conv = False):
    '''
    Initialization of DQN
    :param n_states: the dimension of state space
    :param n_actions: the dimension of action space
    :param if_conv: whether use the convolution algorithm
    '''
    super(DuelingDQN, self).__init__()
    # print("Initialize the Dueling DQN.")
    self.if_conv = if_conv
    self.n_states = n_states
    self.initialization = initialization
    if if_conv:
      # print("Convolution Duealing DQN can only be used for CartPole-v1 environment!")
      self.network = nn.Sequential(
        nn.Conv2d(1, 32, kernel_size=(8, 8), stride=4),
        nn.SiLU(),
        # nn.Conv2d(32, 64, kernel_size=(4, 4), stride=2),
        # nn.SiLU(),
        # nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1),
        # nn.SiLU(),
        nn.Flatten(),
      )

      self.value_func = nn.Sequential(
        nn.Linear(7200, 1))

      self.advantage_func = nn.Sequential(
        nn.Linear(7200, n_actions))
    else:
      self.network = nn.Sequential(
              nn.Linear(n_states, 16),
              nn.SiLU(),
              nn.Linear(16, 16),
              nn.SiLU())

      self.value_func = nn.Sequential(
              nn.Linear(16, 1))

      self.advantage_func = nn.Sequential(
              nn.Linear(16, n_actions))

    self.network.apply(self.initialize_weights)
    self.value_func.apply(self.initialize_weights)
    self.advantage_func.apply(self.initialize_weights)

  def initialize_weights(self, module):
    if isinstance(module, nn.Linear):
      self.initialization(module.weight)
      ## always initialize bias as 0
      nn.init.zeros_(module.bias)
    if isinstance(module, nn.Conv2d):
      self.initialization(module.weight)
      ## always initialize bias as 0
      nn.init.zeros_(module.bias)

  def forward(self, state):
    """
    calculate the Q-value of given state s
    :param state: s, size: n_states
    :return: Q-value, size: n_actions
    """
    if self.if_conv:
      state = state.reshape((-1, 1, self.n_states, 1))
      state = state.repeat((1, 1, 16, 64))
    intermediate = self.network(state)
    adv = self.advantage_func(intermediate)
    return self.value_func(intermediate) + (adv - adv.mean())

class DQNAgent():
  def __init__(self, n_states, n_actions, config):
    '''
    :param n_states: size of states space
    :param n_actions: suze if action space
    :param config: dictionary, stores all value of relevant hyper-parameters
    '''
    self.gamma = config['gamma']                       # discount rate
    self.learning_rate = config['alpha']               # learning rate
    self.factor = config['factor']                     # exploration factor
    self.pool = deque(maxlen=config['capacity'])       # capacity of memory pool
    self.n_states = n_states                           # dimension of states space
    self.n_actions = n_actions                         # dimension of actions space
    self.batch_size = config['batch_size']             # batch size
    self.device = torch.device(config['device'])       # device for calculation
    self.type = torch.float32                          # type of tensor
    self.target = config['target']                     # whether to use target network
    self.replay = config['replay']                     # whether to use  experience replay
    self.N = np.ones((self.n_actions))                 # for UCB exploration policy
    self.policy = config['exploration']                # exploration policy
    # create Deep Q Network
    self.create_network(layers=config['layers'], neurons=config['neurons'], activation=config['activation'], initialization=config['initialization'], if_conv=config['if_conv'])

  def create_network(self, layers=1, neurons=128, activation=nn.ReLU(), initialization=nn.init.normal_, if_conv=False):
    self.Q = DQN(self.n_states, self.n_actions, layers, neurons, activation, initialization).to(self.device)
    if self.target:
      self.target_Q = DQN(self.n_states, self.n_actions, layers, neurons, activation, initialization).to(self.device)
      self.target_Q.load_state_dict(self.Q.state_dict())
    self.criterion = nn.MSELoss()
    self.optimizer = torch.optim.Adam(self.Q.parameters(), lr=self.learning_rate)

  def update_target(self, tau=0):
    '''
    update the parameters of target network
    :param tau: parameter of soft update target network, when tau = 1, means hard update
    :return:
    '''
    if not self.target:
      # if target network isn't used, just return nothing
      return
    if tau == 1:
      # if tau == 1, hard update
      self.target_Q.load_state_dict(self.Q.state_dict())
    else:
      # soft update
      target_param = self.target_Q.state_dict()
      param = self.Q.state_dict()
      for layer_name in param:
          # update parameters of different layers of target network
          target_param[layer_name] = tau * param[layer_name] + (1. - tau) * target_param[layer_name]
      self.target_Q.load_state_dict(target_param)

  def UCB(self, state, t):
    c = self.factor
    with torch.no_grad():
      ucb_values = self.Q(state).cpu().numpy() + c * np.sqrt(2 * np.log(t) / self.N)
      self.N[np.argmax(ucb_values)] += 1
    return np.argmax(ucb_values)

  def e_greedy(self, state):
    if np.random.random() < self.factor:
      return np.random.choice(self.n_actions)
    else:
      with torch.no_grad():
        return np.argmax(self.Q(state).cpu().numpy())

  def boltzmann(self, state):
    with torch.no_grad():
      act_distr = softmax(self.Q(state).cpu().numpy(), self.factor)
      act_distr = act_distr.reshape((-1, ))
    return np.random.choice(self.n_actions, p=act_distr)

  def annealing_factor(self, curr):
    '''
    Annealing the exploration factor
    :param curr: the index of current timestep
    :return:
    '''
    if "greedy" in self.policy:
      self.factor = linear_anneal(curr, 1000, 1, 0, 0.7)
    elif "softmax" in self.policy:
      self.factor = linear_anneal(curr, 1000, 10, 0.0001, 0.7)
    else:
      raise KeyError("Only e-greedy and boltzmann exploration policy can use annealing method!")

  def select_action(self, state, t=0):
    """
    select action by given policy (default: epsilon-greedy)
    :param state: current state
    :param t: current time step
    :param c: confidence level of UCB
    :return: a: action size: 1
    """
    state = torch.from_numpy(state).to(self.device)
    if self.policy == "ucb":
      return self.UCB(state, t)
    elif self.policy == "e-greedy":
      return self.e_greedy(state)
    elif self.policy == "softmax":
      return self.boltzmann(state)
    else:
      raise KeyError("Note: The exploration policy must be 'ucb', 'e-greedy' or 'softmax'")

  def add_memory(self, state, action, reward, next_state, terminal):
    """
    :param state: state of current timestep
    :param action: action of current timestep
    :param reward: reward of current timestep
    :param next_state: next state of current timestep
    :param terminal: whether it is terminated
    :return:
    """
    self.pool.append((state, action, reward, next_state, terminal))

  def sample(self, n):
    """
    sample training data from memory pool
    :param n: the batch size
    :return: states: n * n_states, action: n * 1, reward: n * 1, next_state: n * n_states, terminal: n * 1
    """
    states = []
    actions = []
    rewards = []
    next_states = []
    terminal = []
    sample = random.sample(self.pool, n)
    for pair in sample:
      states.append(pair[0])
      actions.append(pair[1])
      rewards.append(pair[2])
      next_states.append(pair[3])
      terminal.append(pair[4])
    return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(terminal)

  def get_pool_size(self):
    return len(self.pool)

  def train(self):
    """
    train the network
    :return:
    """
    if self.replay:
      # if experience replay, sample training data from experience memory pool
      if self.get_pool_size() < self.batch_size:
        return 0
      states, actions, rewards, next_states, terminal = self.sample(self.batch_size)
    else:
      # else, use the transition of current timestep
      states, actions, rewards, next_states, terminal = self.pool.pop()

      # change the shape of state
      states = np.array(states.reshape((1, -1)))
      actions = np.array(actions)
      rewards = np.array(rewards)
      next_states = np.array(next_states.reshape((1, -1)))
      terminal = np.array(terminal)

    # convert vector to tensor
    states = torch.from_numpy(states).to(self.device)
    actions = torch.from_numpy(actions.reshape((-1, 1))).to(self.device)
    rewards = torch.tensor(rewards.reshape((-1, 1)), dtype=torch.float32).to(self.device)
    next_states = torch.from_numpy(next_states).to(self.device)

    reverse_terminal = terminal ^ 1
    reverse_terminal = torch.from_numpy(reverse_terminal.reshape((-1, 1))).to(self.device)
    q = self.Q(states).gather(1, actions)
    if self.target:
      y = rewards + (self.gamma * self.target_Q(next_states).max(1).values.reshape((-1, 1))).mul_(reverse_terminal)
    else:
      y = rewards + (self.gamma * self.Q(next_states).max(1).values.reshape((-1, 1))).mul_(reverse_terminal)
    self.optimizer.zero_grad()
    loss = self.criterion(q, y)
    loss.backward()
    self.optimizer.step()


class DDQNAgent(DQNAgent):
  def __init__(self, n_states, n_actions, config):
    super(DDQNAgent, self).__init__(n_states, n_actions, config)

  def train(self):
    """
    training the network
    :return:
    """
    if self.replay:
      # if experience replay, sample training data from experience memory pool
      if self.get_pool_size() < self.batch_size:
        return 0
      states, actions, rewards, next_states, terminal = self.sample(self.batch_size)
    else:
      # else, use the transition of current timestep
      states, actions, rewards, next_states, terminal = self.pool.pop()

      # change the shape of state
      states = np.array(states.reshape((1, -1)))
      actions = np.array(actions)
      rewards = np.array(rewards)
      next_states = np.array(next_states.reshape((1, -1)))
      terminal = np.array(terminal)

    states = torch.from_numpy(states).to(self.device)
    actions = torch.from_numpy(actions.reshape((-1, 1))).to(self.device)
    rewards = torch.tensor(rewards.reshape((-1, 1)), dtype=torch.float32).to(self.device)
    next_states = torch.from_numpy(next_states).to(self.device)

    reverse_terminal = terminal ^ 1
    reverse_terminal = torch.from_numpy(reverse_terminal.reshape((-1, 1))).to(self.device)
    q = self.Q(states).gather(1, actions)

    # the main difference from the DQNAgent is the way to calculate the expected target
    expected_actions = self.Q(next_states).argmax(dim=1, keepdim=True)
    y = rewards + (self.gamma * self.target_Q(next_states).gather(1, expected_actions)).mul_(reverse_terminal)


    self.optimizer.zero_grad()
    loss = self.criterion(q, y)
    loss.backward()
    self.optimizer.step()


class DuelingDQNAgent(DDQNAgent):
  def __init__(self, n_states, n_actions, config):
    super(DuelingDQNAgent, self).__init__(n_states, n_actions, config)


  def create_network(self, layers=1, neurons=128, activation=nn.ReLU(), initialization=nn.init.normal_, if_conv=False):
    self.Q = DuelingDQN(self.n_states, self.n_actions, initialization=initialization, if_conv=if_conv).to(self.device)


    if self.target:
      self.target_Q = DuelingDQN(self.n_states, self.n_actions, initialization=initialization, if_conv=if_conv).to(self.device)
      self.target_Q.load_state_dict(self.Q.state_dict())
    self.criterion = nn.MSELoss()
    self.optimizer = torch.optim.Adam(self.Q.parameters(), lr=self.learning_rate)


def run(env, config, n_repetitions=50, n_episodes=1000):
  '''
  :param env: environment of openAI gym
  :param agent: agent of reinforcement learning
  :param update_policy: hard update or soft update
  :param update_para: parameter of update policy, for soft it's tau, for hard, it's how many steps to update target network
  :return: average rewards and the std error of average rewards
  '''

  update_policy = config['update_policy']
  update_para = config['update_para']
  annealing = config['annealing']
  type = config['type']


  n_repetitions = n_repetitions
  n_episodes = n_episodes

  ## Get the dimensions of states and actions
  N_STATES = len(env.reset()[0])
  N_ACTIONS = env.action_space.n


  reward_results = np.empty([n_repetitions, n_episodes])
  for rep in tqdm.trange(n_repetitions):
    if type == 'DQN':
      agent = DQNAgent(N_STATES, N_ACTIONS, config=config)
    elif type == 'DDQN':
      agent = DDQNAgent(N_STATES, N_ACTIONS, config=config)
    elif type == 'DuelingDQN':
      agent = DuelingDQNAgent(N_STATES, N_ACTIONS, config=config)
    else:
      raise KeyError("Agent type must be DQN, DDQN or DuelingDQN!")
    reward_list = []
    steps = 1
    for i in range(n_episodes):
      if annealing:
        agent.annealing_factor(i)
      state, _ = env.reset()
      terminal = False
      overall_reward = 0
      while not terminal:
        action = agent.select_action(state, steps)
        next_state, reward, terminal, truncated, info = env.step(action)
        overall_reward += reward
        agent.add_memory(state, action, reward, next_state, terminal)
        agent.train()
        steps += 1
        if update_policy == 'soft':
          # soft update
          agent.update_target(tau=update_para)
          # print(steps, ": Soft ")
        if update_policy == 'hard':
          # hard update
          if steps % update_para == 0:
            agent.update_target(tau=1)
        state = next_state
        if truncated:
          break
      reward_list.append(overall_reward)
      # print(overall_reward)
    reward_results[rep] = reward_list
  mean_reward = np.mean(reward_results, axis=0)
  std_reward = np.std(reward_results, axis=0)

  return mean_reward, std_reward