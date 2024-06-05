import numpy as np
import random
from environment import Env
from collections import defaultdict

class QLearningAgent:
    def __init__(self, actions):
        self.actions = actions
        self.q_table = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])

    def learn(self, state, action, reward, next_state, learning_rate, discount_factor):
        current_q = self.q_table[state][action]
        new_q = reward + discount_factor * max(self.q_table[next_state])
        self.q_table[state][action] += learning_rate * (new_q - current_q)

    def get_action(self, state, epsilon):
        if np.random.rand() < epsilon:
            action = np.random.choice(self.actions)
        else:
            state_action = self.q_table[state]
            action = self.arg_max(state_action)
        return action

    @staticmethod
    def arg_max(state_action):
        max_index_list = []
        max_value = state_action[0]
        for index, value in enumerate(state_action):
            if value > max_value:
                max_index_list.clear()
                max_value = value
                max_index_list.append(index)
            elif value == max_value:
                max_index_list.append(index)
        return random.choice(max_index_list)

# Hyperparameter Tuning
env = Env()
learning_rates = [0.01, 0.05, 0.1]
discount_factors = [0.5, 0.7, 0.9]
epsilons = [0.05, 0.1, 0.2]

best_reward = float('-inf')
best_hyperparameters = None

for lr in learning_rates:
    for df in discount_factors:
        for eps in epsilons:
            agent = QLearningAgent(actions=list(range(env.n_actions)))

            total_rewards = []
            for episode in range(1000):
                state = env.reset()
                episode_reward = 0
                while True:
                    action = agent.get_action(str(state), eps)
                    next_state, reward, done = env.step(action)
                    agent.learn(str(state), action, reward, str(next_state), lr, df)
                    env.print_value_all(agent.q_table)
                    state = next_state
                    episode_reward += reward
                    if done:
                        break
                total_rewards.append(episode_reward)

            avg_reward = np.mean(total_rewards)
            if avg_reward > best_reward:
                best_reward = avg_reward
                best_hyperparameters = (lr, df, eps)

print("Best hyperparameters:", best_hyperparameters)
print("Best average reward:", best_reward)
