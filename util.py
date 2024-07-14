import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


class LearningCurvePlot:

    def __init__(self, title=None):
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlabel('Episode')
        self.ax.set_ylabel('Mean Reward')
        if title is not None:
            self.ax.set_title(title)

    def add_curve(self, y, std, label=None):
        ''' y: vector of average reward results
        label: string to appear as label in plot legend '''
        steps = np.arange(y.shape[0])
        y = smooth(y, window=21)
        std = smooth(std, window=21)

        if label is not None:
            self.ax.plot(y, label=label)
            self.ax.fill_between(steps, y - std, y + std, alpha=0.3)
        else:
            self.ax.plot(y)
            self.ax.fill_between(steps, y - std, y + std, alpha=0.3)

    def set_ylim(self, lower, upper):
        self.ax.set_ylim([lower, upper])

    def add_hline(self, height, label):
        self.ax.axhline(height, ls='--', c='k', label=label)

    def save(self, name='test.png'):
        ''' name: string for filename of saved figure '''
        self.ax.legend()
        self.fig.savefig(name, dpi=300)


class LearningCurvePlotNoError:

    def __init__(self, title=None):
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlabel('Episode')
        self.ax.set_ylabel('Mean Reward')
        if title is not None:
            self.ax.set_title(title)

    def add_curve(self, y, std, label=None):
        ''' y: vector of average reward results
        label: string to appear as label in plot legend '''
        steps = np.arange(y.shape[0])
        y = smooth(y, window=11)
        std = smooth(std, window=11)

        if label is not None:
            self.ax.plot(y, label=label)
        else:
            self.ax.plot(y)

    def set_ylim(self, lower, upper):
        self.ax.set_ylim([lower, upper])

    def add_hline(self, height, label):
        self.ax.axhline(height, ls='--', c='k', label=label)

    def save(self, name='test.png'):
        ''' name: string for filename of saved figure '''
        self.ax.legend()
        self.fig.savefig(name, dpi=300)


def smooth(y, window, poly=1):
    '''
    y: vector to be smoothed
    window: size of the smoothing window '''
    return savgol_filter(y, window, poly)


def softmax(x, temp):
    ''' Computes the softmax of vector x with temperature parameter 'temp' '''
    x = x / temp  # scale by temperature
    z = x - max(x)  # substract max to prevent overflow of softmax
    return np.exp(z) / np.sum(np.exp(z))  # compute softmax


def linear_anneal(t,T,start,final,percentage):
    ''' Linear annealing scheduler
    t: current timestep
    T: total timesteps
    start: initial value
    final: value after percentage*T steps
    percentage: percentage of T after which annealing finishes
    '''
    final_from_T = int(percentage*T)
    if t > final_from_T:
        return final
    else:
        return final + (start - final) * (final_from_T - t)/final_from_T
