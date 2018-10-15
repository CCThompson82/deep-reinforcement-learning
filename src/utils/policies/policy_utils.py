"""Utils for policy action selection"""

import numpy as np


def get_greedy_epsilon_action(Q, state, epsilon, nb_A):
    if np.random.rand() < epsilon:
        action = np.random.randint(0, nb_A)
    else:
        action = np.argmax(Q[state])
    return action


def get_greedy_action(Q, state):
    return np.argmax(Q[state])


def get_sigmoid_epsilon(episode_it, decay_rate=0.0001, x50=10000, floor=0.05, ceil=0.5):
    epsilon = ceil / (1 + np.exp(decay_rate*(episode_it - x50)))
    return np.max([epsilon, floor])


def get_inverse_episode_epsilon(episode_it, floor=0.0):
    return np.max([1.0 / episode_it, floor])
