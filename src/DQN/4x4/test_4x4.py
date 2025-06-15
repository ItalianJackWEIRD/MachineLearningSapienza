import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from frozenlake_dqn_model_4x4 import DQN
import tensorflow as tf
from config4x4 import CONFIG

def state_to_input(state, num_states):
    one_hot = np.zeros(num_states)
    one_hot[state] = 1
    return np.expand_dims(one_hot, axis=0)

def test(episodes=10):
    env = gym.make(CONFIG['env_name'], map_name=CONFIG['map_name'], is_slippery=CONFIG['is_slippery'], render_mode='human')
    num_states = env.observation_space.n
    num_actions = env.action_space.n

    hidden1 = CONFIG['hidden_layer_sizes'][0]
    dqn = DQN(num_states, hidden1, num_actions)
    dqn.load_weights("frozen_lake_dql_4x4.weights.h5")

    rewards = []
    steps = []

    for episode in range(episodes):
        state = env.reset()[0]
        done = False
        total_reward = 0
        step_count = 0

        while not done:
            q_values = dqn(state_to_input(state, num_states))
            action = tf.argmax(q_values[0]).numpy()
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            total_reward += reward
            step_count += 1

        rewards.append(total_reward)
        steps.append(step_count)
        print(f"Episode {episode + 1}: Reward = {total_reward}, Steps = {step_count}")

    env.close()

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, episodes + 1), rewards, marker='o')
    plt.title('Rewards per Episode (Test)')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(range(1, episodes + 1), steps, marker='o', color='orange')
    plt.title('Steps per Episode (Test)')
    plt.xlabel('Episode')
    plt.ylabel('Number of Steps')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("frozen_lake_test_results_4x4.png")
    plt.show()

    print(f"\nAverage reward over {episodes} episodes: {np.mean(rewards):.2f}")
    print(f"Average steps per episode: {np.mean(steps):.2f}")

if __name__ == '__main__':
    test()
