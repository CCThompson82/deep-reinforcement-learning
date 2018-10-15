"""Functions for running sarsa related episodes"""

import numpy as np
from src.utils.policies import policy_utils as policy


def run_sarsa_episode(env, Q, epsilon, alpha, gamma):
    state_0 = env.reset()
    action_0 = policy.get_greedy_epsilon_action(Q, state_0, epsilon, env.nA)

    while True:
        Q, state_1, action_1, terminal_status, info = generate_sarsa_step(
            env, Q, state_0, action_0, epsilon, alpha, gamma)

        state_0 = state_1
        action_0 = action_1
        if terminal_status:
            break
    return Q


def generate_sarsa_step(env, Q, state_0, action_0, epsilon, alpha, gamma):
    """Generates a single step of a sarsa episode

    Args
        env (openAI environment)
        Q (collections.defaultdict): Q table
        epsilon (float): random action rate
        alpha (float): learning rate
        gamma (float): discount rate

    """
    state_1, reward0, terminal_status, info = env.step(a=action_0)

    action_1 = policy.get_greedy_epsilon_action(Q, state_1, epsilon, env.nA)

    G1 = Q[state_1][action_1]

    # update Q
    current_value = Q[state_0][action_0]
    discounted_return = G1 * gamma
    delta_Q = (reward0 + discounted_return - current_value)

    Q[state_0][action_0] = current_value + (alpha * delta_Q)

    return Q, state_1, action_1, terminal_status, info


def run_sarsamax_episode(env, Q, epsilon, alpha, gamma):
    state_0 = env.reset()
    action_0 = policy.get_greedy_action(Q, state_0)

    while True:
        Q, state_1, action_1, episode_finished, info = \
            generate_sarsamax_step(
                env, Q, state_0, action_0, epsilon, alpha, gamma)

        state_0 = state_1
        action_0 = action_1

        if episode_finished:
            break
    return Q


def generate_sarsamax_step(env, Q, state_0, action_0, epsilon, alpha, gamma):
    state_1, reward0, episode_finished, info = env.step(a=action_0)
    # evaluate the greediest action
    action_1 = policy.get_greedy_action(Q, state_1)

    G1 = Q[state_1][action_1]

    current_value_estimate = Q[state_0][action_0]
    discounted_return = G1 * gamma
    delta_Q = (reward0 + discounted_return - current_value_estimate)
    Q[state_0][action_0] = current_value_estimate + (alpha * delta_Q)

    # but provide the epsilon greedy action
    action_1 = policy.get_greedy_epsilon_action(Q, state_1, epsilon, env.nA)

    return Q, state_1, action_1, episode_finished, info


def run_expected_sarsa_episode(env, Q, epsilon, alpha, gamma):
    state_0 = env.reset()
    action_0 = policy.get_greedy_epsilon_action(Q, state_0, epsilon, env.nA)

    while True:

        Q, state_1, action_1, episode_finished, info = \
            generate_expected_sarsa_step(
                env, Q, state_0, action_0, epsilon, alpha, gamma)

        state_0 = state_1
        action_0 = action_1

        if episode_finished:
            break

    return Q


def generate_expected_sarsa_step(env, Q, state_0, action_0, epsilon, alpha, gamma):
    state_1, reward0, episode_finished, info = env.step(action_0)

    expected_sarsa_update_step(Q, action_0, alpha, epsilon, gamma, reward0, state_0, state_1, nb_A=env.nA)

    # choose an action here, even though not important for SA value estimation
    action_1 = policy.get_greedy_epsilon_action(Q, state_1, epsilon, env.nA)

    return Q, state_1, action_1, episode_finished, info


def expected_sarsa_update_step(Q, action_0, alpha, epsilon, gamma, reward0, state_0, state_1, nb_A):
    state_action_values = Q[state_1]
    max_action = np.argmax(state_action_values)
    action_probabilities = np.full(nb_A, epsilon / nb_A)
    action_probabilities[max_action] += 1 - epsilon
    G1 = np.sum(
        [sa_value * action_prob for sa_value, action_prob in
         zip(state_action_values, action_probabilities)])
    current_value_estimate = Q[state_0][action_0]
    discounted_return = G1 * gamma
    delta_Q = (reward0 + discounted_return - current_value_estimate)
    Q[state_0][action_0] = current_value_estimate + (alpha * delta_Q)
    return Q