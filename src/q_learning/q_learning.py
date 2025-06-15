import numpy as np

class QLearningAgent:
    def __init__(self, state_size, action_size, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.05):
        self.q_table = np.zeros((state_size, action_size))  # Initialize Q-table 16 x 4 for FrozenLake
        self.alpha = alpha  # learning rate 
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.action_size = action_size

    def choose_action(self, state):   # 0=left, 1=down, 2=right, 3=up       
        # Epsilon-greedy action selection
        # With probability epsilon, choose a random action; otherwise, choose the best action based on Q-values
        # This encourages exploration in the early episodes and exploitation later on
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        return np.argmax(self.q_table[state])

    def update(self, state, action, reward, next_state, done):
        best_next = np.max(self.q_table[next_state])
        target = reward + self.gamma * best_next * (not done)
        self.q_table[state, action] += self.alpha * (target - self.q_table[state, action])
        if done:
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min) # Decay Formula Alt "(epsilon - epsilon_decay, 0)" decay = 0.001 or smaller
