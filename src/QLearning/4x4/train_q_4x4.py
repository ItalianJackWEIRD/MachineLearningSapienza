import gymnasium as gym
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from q_learning import QLearningAgent
from utils import plot_rewards, plot_smoothed_rewards
from config_q_4x4 import CONFIG



env = gym.make(CONFIG['env_name'], map_name=CONFIG['map_name'], is_slippery=CONFIG['is_slippery'])
agent = QLearningAgent(
    state_size=env.observation_space.n,
    action_size=env.action_space.n,
    alpha=CONFIG['alpha'],
    gamma=CONFIG['gamma'],
    epsilon=CONFIG['epsilon_start'],
    epsilon_decay=CONFIG['epsilon_decay'],
    epsilon_min=CONFIG['epsilon_min']
)

episodes = CONFIG['episodes']
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

current_dir = os.path.dirname(__file__)
plot_rewards(rewards, title='Tabular Q-Learning on FrozenLake 4x4', save_dir=current_dir)
plot_smoothed_rewards(rewards, title='Smoothed Rewards on FrozenLake 4x4', save_dir=current_dir)

successes = [int(r > 0) for r in rewards]
rolling_success = pd.Series(successes).rolling(window=100).mean()

sns.lineplot(x=range(len(rolling_success)), y=rolling_success)
plt.title("Rolling Success Rate (window=100)")
plt.xlabel("Episode")
plt.ylabel("Success Rate")
plt.tight_layout()
plt.savefig(os.path.join(current_dir,"Rolling_Success_Rate_4x4.png"))
plt.close()

plt.figure()
plt.plot(range(len(epsilons)), epsilons)
plt.title("Epsilon Decay Over Episodes")
plt.xlabel("Episode")
plt.ylabel("Epsilon (Exploration Rate)")
plt.tight_layout()
plt.savefig(os.path.join(current_dir,"Epsilon_Decay_4x4.png"))
plt.close()

np.save(os.path.join(current_dir,"q_table_4x4.npy"), agent.q_table)
