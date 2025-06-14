import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import random

# Define model
class DQN(tf.keras.Model):
    def __init__(self, in_states, h1_nodes, out_actions):
        super(DQN, self).__init__()
        self.fc1 = layers.Dense(h1_nodes, activation='relu')
        self.out = layers.Dense(out_actions)

    def call(self, x):
        return self.out(self.fc1(x))

# Replay memory
class ReplayMemory:
    def __init__(self, maxlen):
        self.memory = []
        self.maxlen = maxlen

    def append(self, transition):
        if len(self.memory) >= self.maxlen:
            self.memory.pop(0)
        self.memory.append(transition)

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)

    def __len__(self):
        return len(self.memory)