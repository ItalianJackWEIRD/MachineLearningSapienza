import tensorflow as tf
from tensorflow.keras import layers

class DQN(tf.keras.Model):
    def __init__(self, num_states, hidden1, num_actions):
        super(DQN, self).__init__()
        self.dense1 = layers.Dense(hidden1, activation='relu')
        self.output_layer = layers.Dense(num_actions, activation='linear')

    def call(self, inputs):
        x = self.dense1(inputs)
        return self.output_layer(x)

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def append(self, transition):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        import random
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
