import numpy as np
from collections import defaultdict
from src.utils.policies import policy_utils
from src.utils.episodes import sarsa as sarsa_utils

class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.episodes = 0

        self.epsilon_config = {'decay_rate': 1e-4,
                               'inflection_episode': 5000,
                               'minimum_epsilon': 0.01,
                               'ceiling_epsilon': 1.0}
        self.alpha = 1e-3
        self.gamma = 0.9

    @property
    def epsilon(self):
        epsilon = policy_utils.get_sigmoid_epsilon(
            episode_it=self.episodes,
            decay_rate=self.epsilon_config['decay_rate'],
            x50=self.epsilon_config['inflection_episode'],
            floor=self.epsilon_config['minimum_epsilon'],
            ceil=self.epsilon_config['ceiling_epsilon'])
        return epsilon

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """

        action = policy_utils.get_greedy_epsilon_action(
            Q=self.Q, state=state, epsilon=self.epsilon, nb_A=self.nA)

        return action

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        self.Q = sarsa_utils.expected_sarsa_update_step(Q=self.Q,
                                                        state_0=state,
                                                        action_0=action,
                                                        state_1=next_state,
                                                        reward0=reward,
                                                        epsilon=self.epsilon,
                                                        alpha=self.alpha,
                                                        gamma=self.gamma,
                                                        nb_A=self.nA)

