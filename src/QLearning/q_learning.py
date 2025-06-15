import numpy as np

class QLearningAgent:
    def __init__(self, state_size, action_size, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.05):
        self.q_table = np.zeros((state_size, action_size))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.action_size = action_size

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        return np.argmax(self.q_table[state])

    def update(self, state, action, reward, next_state, done):
        best_next = np.max(self.q_table[next_state])
        target = reward + self.gamma * best_next * (not done)
        self.q_table[state, action] += self.alpha * (target - self.q_table[state, action])
        if done:
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
