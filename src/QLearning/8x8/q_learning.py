import numpy as np

class QLearningAgent:
    def __init__(self, state_size, action_size, config):
        self.q_table = np.zeros((state_size, action_size))
        self.alpha = config['alpha']
        self.gamma = config['gamma']
        self.epsilon = config['epsilon_start']
        self.epsilon_decay = config['epsilon_decay']
        self.epsilon_min = config['epsilon_min']
        self.action_size = action_size

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        return np.argmax(self.q_table[state])

    def update(self, state, action, reward, next_state, done):
        if done and reward == 0:
            shaped_reward = -1
        elif reward == 0:
            shaped_reward = -0.01
        else:
            shaped_reward = reward
        best_next = np.max(self.q_table[next_state])
        target = shaped_reward + self.gamma * best_next * (not done)
        self.q_table[state, action] += self.alpha * (target - self.q_table[state, action])
        if done:
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
