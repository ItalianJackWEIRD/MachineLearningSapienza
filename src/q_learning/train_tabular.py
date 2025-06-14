import gymnasium as gym
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from q_learning import QLearningAgent
from utils import plot_rewards, plot_smoothed_rewards

env = gym.make('FrozenLake-v1', is_slippery=True)
agent = QLearningAgent(state_size=env.observation_space.n, action_size=env.action_space.n)

episodes = 10000
rewards = []
epsilons = []

for ep in range(episodes):
    state, _ = env.reset()
    total_reward = 0
    done = False

    while not done:
        action = agent.choose_action(state)
        next_state, reward, done, truncated, _ = env.step(action)
        agent.update(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

    rewards.append(total_reward)
    epsilons.append(agent.epsilon)

plot_rewards(rewards, title='Tabular Q-Learning on FrozenLake')
plot_smoothed_rewards(rewards, title='Smoothed Rewards on FrozenLake')

successes = [int(r > 0) for r in rewards]
rolling_success = pd.Series(successes).rolling(window=100).mean()

sns.lineplot(x=range(len(rolling_success)), y=rolling_success)
plt.title("Rolling Success Rate (window=100)")
plt.xlabel("Episode")
plt.ylabel("Success Rate")
plt.tight_layout()
plt.savefig("Rolling_Success_Rate.png")
plt.close()

plt.figure()
plt.plot(range(len(epsilons)), epsilons)
plt.title("Epsilon Decay Over Episodes")
plt.xlabel("Episode")
plt.ylabel("Epsilon (Exploration Rate)")
plt.tight_layout()
plt.savefig("Epsilon_Decay.png")
plt.close()

np.save("q_table.npy", agent.q_table)
