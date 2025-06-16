import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from frozenlake_dqn_model_4x4 import DQN, ReplayMemory
import tensorflow as tf
import random
import os
from config4x4 import CONFIG

def state_to_input(state, num_states):
    one_hot = np.zeros(num_states)
    one_hot[state] = 1
    return np.expand_dims(one_hot, axis=0)

def train():
    env = gym.make(CONFIG['env_name'], map_name=CONFIG['map_name'], is_slippery=CONFIG['is_slippery'])
    num_states = env.observation_space.n
    num_actions = env.action_space.n

    hidden1 = CONFIG['hidden_layer_sizes'][0]

    policy_dqn = DQN(num_states, hidden1, num_actions)
    policy_dqn.build(input_shape=(None, num_states))

    target_dqn = DQN(num_states, hidden1, num_actions)
    target_dqn.set_weights(policy_dqn.get_weights())

    optimizer = tf.keras.optimizers.Adam(learning_rate=CONFIG['learning_rate'])
    loss_fn = tf.keras.losses.MeanSquaredError()

    memory = ReplayMemory(CONFIG['memory_size'])
    mini_batch_size = CONFIG['mini_batch_size']
    gamma = CONFIG['gamma']
    sync_rate = CONFIG['sync_rate']
    epsilon = CONFIG['epsilon_start']
    epsilon_min = CONFIG['epsilon_min']
    epsilon_decay = CONFIG['epsilon_decay']
    episodes = CONFIG['episodes']

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

            epsilon = max(epsilon * epsilon_decay, epsilon_min)
            epsilon_history.append(epsilon)

            if step_count > sync_rate:
                target_dqn.set_weights(policy_dqn.get_weights())
                step_count = 0

        if (i+1) % 500 == 0 or i == episodes - 1:
            avg_reward = np.mean(rewards_per_episode[max(0,i-499):i+1])
            print(f"Episode {i+1}/{episodes} - Epsilon: {epsilon:.3f} - Recent avg reward: {avg_reward:.3f}")


    current_dir = os.path.dirname(__file__)

    policy_dqn.save_weights(os.path.join(current_dir, "frozen_lake_dql_4x4.weights.h5"))

    plt.figure(figsize=(12, 5))
    window_size = 100
    cumulative_rewards = [np.sum(rewards_per_episode[max(0, i - window_size + 1):(i + 1)]) for i in range(len(rewards_per_episode))]

    plt.subplot(1, 2, 1)
    plt.plot(cumulative_rewards, label='Cumulative Reward (100 eps)')
    plt.title('Training Performance on FrozenLake 4x4')
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward (Last 100 Episodes)')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epsilon_history, label='Epsilon (Exploration Rate)', color='orange')
    plt.title('Epsilon Decay over Training')
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(current_dir,"frozen_lake_dql_4x4.png"))
    plt.show()

if __name__ == '__main__':
    train()
