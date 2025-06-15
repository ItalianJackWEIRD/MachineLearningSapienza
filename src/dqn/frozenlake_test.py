import gymnasium as gym
import numpy as np
from frozenlake_dqn_model import DQN
import tensorflow as tf

ACTIONS = ['L','D','R','U']

def state_to_input(state, num_states):
    one_hot = np.zeros(num_states)
    one_hot[state] = 1
    return np.expand_dims(one_hot, axis=0)

def test(episodes=10, is_slippery=False):
    env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=is_slippery, render_mode='human')
    num_states = env.observation_space.n
    num_actions = env.action_space.n

    dqn = DQN(num_states, num_states, num_actions)
    dqn.load_weights("frozen_lake_dql.h5")

    for _ in range(episodes):
        state = env.reset()[0]
        done = False
        while not done:
            q_values = dqn(state_to_input(state, num_states))
            action = tf.argmax(q_values[0]).numpy()
            state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
    env.close()

if __name__ == '__main__':
    test()
