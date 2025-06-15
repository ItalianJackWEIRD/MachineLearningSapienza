import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from frozenlake_dqn_model import DQN, ReplayMemory
import tensorflow as tf
import random

ACTIONS = ['L','D','R','U']

def state_to_input(state, num_states):
    one_hot = np.zeros(num_states)
    one_hot[state] = 1
    return np.expand_dims(one_hot, axis=0)

def train(episodes=1000, is_slippery=False):
    env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=is_slippery)
    num_states = env.observation_space.n
    num_actions = env.action_space.n

    policy_dqn = DQN(num_states, num_states, num_actions)
    target_dqn = DQN(num_states, num_states, num_actions)
    target_dqn.set_weights(policy_dqn.get_weights())

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss_fn = tf.keras.losses.MeanSquaredError()

    memory = ReplayMemory(1000)
    mini_batch_size = 32
    gamma = 0.9
    sync_rate = 10
    epsilon = 1.0
    rewards_per_episode = np.zeros(episodes)
    epsilon_history = []
    step_count = 0

    for i in range(episodes):
        state = env.reset()[0]
        done = False

        while not done:
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                q_values = policy_dqn(state_to_input(state, num_states))
                action = tf.argmax(q_values[0]).numpy()

            new_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            memory.append((state, action, new_state, reward, done))
            state = new_state
            step_count += 1

        if reward == 1:
            rewards_per_episode[i] = 1

        if len(memory) >= mini_batch_size and np.sum(rewards_per_episode) > 0:
            mini_batch = memory.sample(mini_batch_size)
            states, actions, new_states, rewards, dones = zip(*mini_batch)
            
            states = np.vstack([state_to_input(s, num_states) for s in states])
            new_states = np.vstack([state_to_input(s, num_states) for s in new_states])
            actions = np.array(actions)
            rewards = np.array(rewards)
            dones = np.array(dones)

            future_q = target_dqn.predict(new_states).max(axis=1)
            targets = policy_dqn.predict(states)
            for idx, action in enumerate(actions):
                targets[idx, action] = rewards[idx] if dones[idx] else rewards[idx] + gamma * future_q[idx]

            policy_dqn.compile(optimizer=optimizer, loss=loss_fn)
            policy_dqn.fit(states, targets, verbose=0)

            epsilon = max(epsilon - 1/episodes, 0)
            epsilon_history.append(epsilon)

            if step_count > sync_rate:
                target_dqn.set_weights(policy_dqn.get_weights())
                step_count = 0

    policy_dqn.save_weights("frozen_lake_dql.h5")
    plt.figure(1)
    plt.subplot(121)
    plt.plot([np.sum(rewards_per_episode[max(0, x-100):(x+1)]) for x in range(episodes)])
    plt.subplot(122)
    plt.plot(epsilon_history)
    plt.savefig("frozen_lake_dql.png")

if __name__ == '__main__':
    train()