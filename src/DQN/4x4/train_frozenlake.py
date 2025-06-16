import gymnasium as gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random
from collections import deque
from dqn_model import DQN  # Assumes DQN is defined in dqn_model.py

class ReplayMemory:
    def __init__(self, maxlen):
        self.memory = deque(maxlen=maxlen)

    def append(self, transition):
        self.memory.append(transition)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class FrozenLakeTrainer:
    def __init__(self):
        self.learning_rate = 0.001
        self.gamma = 0.9
        self.sync_rate = 10
        self.replay_size = 1000
        self.batch_size = 32
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.loss_fn = tf.keras.losses.MeanSquaredError()
        self.actions = ['L', 'D', 'R', 'U']

    def train(self, episodes=1000, is_slippery=False):
        env = gym.make('FrozenLake-v1', map_name='4x4', is_slippery=is_slippery)
        num_states = env.observation_space.n
        num_actions = env.action_space.n

        memory = ReplayMemory(self.replay_size)
        epsilon = 1.0
        policy_net = DQN(num_states, num_states, num_actions)
        target_net = DQN(num_states, num_states, num_actions)

        policy_net.build((None, num_states))
        target_net.build((None, num_states))
        target_net.set_weights(policy_net.get_weights())

        print("Policy (random, before training):")
        self.print_policy(policy_net, num_states)

        rewards = np.zeros(episodes)
        epsilons = []
        losses = []
        steps = 0

        for ep in range(episodes):
            state = env.reset()[0]
            terminated = truncated = False
            total_loss = 0

            while not terminated and not truncated:
                if random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    q_values = policy_net(self.state_to_input(state, num_states))
                    action = tf.argmax(q_values[0]).numpy()

                next_state, reward, terminated, truncated, _ = env.step(action)
                memory.append((state, action, next_state, reward, terminated))
                state = next_state
                steps += 1

            if reward == 1:
                rewards[ep] = 1

            if len(memory) > self.batch_size and np.sum(rewards) > 0:
                batch = memory.sample(self.batch_size)
                batch_loss = self.optimize(policy_net, target_net, batch, num_states)
                losses.append(batch_loss)

                epsilon = max(epsilon - 1 / episodes, 0)
                epsilons.append(epsilon)

                if steps > self.sync_rate:
                    target_net.set_weights(policy_net.get_weights())
                    steps = 0

        env.close()
        policy_net.save_weights("frozen_lake_dql_tf.weights.h5")

        print("\nPolicy (trained, after training):")
        self.print_policy(policy_net, num_states)

        self.plot_training_results(rewards, epsilons, losses)

    def optimize(self, policy_net, target_net, batch, num_states):
        states, actions, next_states, rewards, dones = zip(*batch)

        states = np.vstack([self.state_to_input(s, num_states) for s in states])
        next_states = np.vstack([self.state_to_input(s, num_states) for s in next_states])
        actions = np.array(actions)
        rewards = np.array(rewards, dtype=np.float32)
        dones = np.array(dones)

        next_q = target_net(next_states).numpy().max(axis=1)
        target = policy_net(states).numpy()

        for i in range(len(batch)):
            target[i, actions[i]] = rewards[i] if dones[i] else rewards[i] + self.gamma * next_q[i]

        with tf.GradientTape() as tape:
            preds = policy_net(states)
            loss = self.loss_fn(target, preds)

        grads = tape.gradient(loss, policy_net.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, policy_net.trainable_variables))

        return loss.numpy()

    def state_to_input(self, state, num_states):
        one_hot = np.zeros((1, num_states))
        one_hot[0, state] = 1
        return one_hot

    def plot_training_results(self, rewards, epsilons, losses):
        window = 100
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')

        # Plot 1: Reward per episode
        plt.figure()
        plt.plot(rewards)
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Reward per Episode")
        plt.grid(True)
        plt.savefig("reward_per_episode.png")
        plt.close()

        # Plot 2: Moving average reward
        plt.figure()
        plt.plot(moving_avg, color='green')
        plt.xlabel("Episode")
        plt.ylabel("Avg Reward")
        plt.title(f"{window}-Episode Moving Average Reward")
        plt.grid(True)
        plt.savefig("moving_average_reward.png")
        plt.close()

        # Plot 3: Epsilon decay
        plt.figure()
        plt.plot(epsilons, color='orange')
        plt.xlabel("Episode")
        plt.ylabel("Epsilon")
        plt.title("Epsilon Decay")
        plt.grid(True)
        plt.savefig("epsilon_decay.png")
        plt.close()

        # Plot 4: Loss over time
        plt.figure()
        plt.plot(losses, color='red')
        plt.xlabel("Training Step")
        plt.ylabel("Loss")
        plt.title("Loss Over Time")
        plt.grid(True)
        plt.savefig("loss_over_time.png")
        plt.close()


    def print_policy(self, net, num_states):
        for s in range(num_states):
            q_vals = net(self.state_to_input(s, num_states))[0].numpy()
            action = self.actions[np.argmax(q_vals)]
            q_str = ' '.join([f'{q:+.2f}' for q in q_vals])
            print(f'{s:02},{action},[{q_str}]', end=' ')
            if (s + 1) % 4 == 0:
                print()

if __name__ == "__main__":
    trainer = FrozenLakeTrainer()
    trainer.train(episodes=10000, is_slippery=True)
