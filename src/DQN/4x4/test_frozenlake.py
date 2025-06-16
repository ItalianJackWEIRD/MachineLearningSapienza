import gymnasium as gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from dqn_model import DQN

class FrozenLakeTester:
    def __init__(self):
        self.actions = ['L', 'D', 'R', 'U']

    def test(self, episodes=100, is_slippery=False):
        env = gym.make('FrozenLake-v1', map_name='4x4', is_slippery=is_slippery, render_mode='human')
        num_states = env.observation_space.n
        num_actions = env.action_space.n

        current_dir = os.path.dirname(__file__)

        policy_net = DQN(num_states, num_states, num_actions)
        policy_net.build((None, num_states))
        policy_net.load_weights(os.path.join(current_dir, "frozen_lake_dql_tf.weights.h5"))

        print("Policy (trained):")
        self.print_policy(policy_net, num_states)

        rewards = []
        steps_per_episode = []

        for ep in range(episodes):
            state = env.reset()[0]
            terminated = truncated = False
            total_reward = 0
            steps = 0

            while not terminated and not truncated:
                q_values = policy_net(self.state_to_input(state, num_states))
                action = tf.argmax(q_values[0]).numpy()
                state, reward, terminated, truncated, _ = env.step(action)

                total_reward += reward
                steps += 1

            rewards.append(total_reward)
            steps_per_episode.append(steps)

        env.close()

        self.plot_test_results(rewards, steps_per_episode)

        success_rate = np.sum(rewards) / episodes * 100
        print(f"\nSuccess rate over {episodes} episodes: {success_rate:.2f}%")

    def state_to_input(self, state, num_states):
        one_hot = np.zeros((1, num_states))
        one_hot[0, state] = 1
        return one_hot

    def print_policy(self, net, num_states):
        for s in range(num_states):
            q_vals = net(self.state_to_input(s, num_states))[0].numpy()
            action = self.actions[np.argmax(q_vals)]
            q_str = ' '.join([f'{q:+.2f}' for q in q_vals])
            print(f'{s:02},{action},[{q_str}]', end=' ')
            if (s + 1) % 4 == 0:
                print()

    def plot_test_results(self, rewards, steps_per_episode):
        episodes = len(rewards)

        current_dir = os.path.dirname(__file__)

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.bar(range(episodes), rewards, color='skyblue')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Reward per Test Episode')
        plt.ylim(0, 1.2)
        plt.grid(True, axis='y')

        plt.subplot(1, 2, 2)
        plt.bar(range(episodes), steps_per_episode, color='salmon')
        plt.xlabel('Episode')
        plt.ylabel('Steps')
        plt.title('Steps per Test Episode')
        plt.grid(True, axis='y')

        plt.tight_layout()
        plt.savefig(os.path.join(current_dir, "test_results.png"))
        plt.show()

if __name__ == "__main__":
    tester = FrozenLakeTester()
    tester.test(episodes=100, is_slippery=False)
