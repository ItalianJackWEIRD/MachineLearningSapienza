import gymnasium as gym
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from q_learning import QLearningAgent
from config_q_4x4 import CONFIG
import os

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

df_rewards = pd.DataFrame({'Episode': range(len(rewards)), 'Reward': rewards})
sns.lineplot(x='Episode', y='Reward', data=df_rewards)
plt.title('Tabular Q-Learning on FrozenLake 4x4')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.tight_layout()
plt.savefig("Tabular_Q-Learning_on_FrozenLake_4x4.png")
plt.close()

successes = [int(r > 0) for r in rewards]
rolling_success = pd.Series(successes).rolling(window=100).mean()
sns.lineplot(x=range(len(rolling_success)), y=rolling_success)
plt.title("Rolling Success Rate (window=100)")
plt.xlabel("Episode")
plt.ylabel("Success Rate")
plt.tight_layout()
plt.savefig("Rolling_Success_Rate_4x4.png")
plt.close()

plt.figure()
plt.plot(range(len(epsilons)), epsilons)
plt.title("Epsilon Decay Over Episodes")
plt.xlabel("Episode")
plt.ylabel("Epsilon (Exploration Rate)")
plt.tight_layout()
plt.savefig("Epsilon_Decay_4x4.png")
plt.close()

np.save("q_table_4x4.npy", agent.q_table)

desc = env.unwrapped.desc.astype(str)
map_size = desc.shape[0]
color_map = {'S': 'green', 'F': 'lightblue', 'H': 'black', 'G': 'red'}

fig, ax = plt.subplots(figsize=(5, 5))
for row in range(map_size):
    for col in range(map_size):
        tile = desc[row][col]
        ax.add_patch(plt.Rectangle((col, map_size - row - 1), 1, 1, color=color_map[tile], ec='gray'))
        ax.text(col + 0.5, map_size - row - 1 + 0.5, tile, ha='center', va='center', color='white' if tile == 'H' else 'black', fontsize=12)

ax.set_xlim(0, map_size)
ax.set_ylim(0, map_size)
ax.set_xticks(np.arange(map_size + 1))
ax.set_yticks(np.arange(map_size + 1))
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_title('FrozenLake Map Layout (4x4)')
plt.grid(True)
plt.gca().set_aspect('equal', adjustable='box')
plt.tight_layout()
plt.savefig("FrozenLake_4x4_Layout.png")
plt.close()

plt.figure(figsize=(10, 8))
ax = sns.heatmap(agent.q_table, annot=True, cmap='viridis', fmt=".2f", cbar_kws={'label': 'Q-value'})

ax.set_title("Q-Table Heatmap (FrozenLake 4x4)", fontsize=14)
ax.set_xlabel("Actions", fontsize=12)
ax.set_ylabel("States", fontsize=12)
ax.set_xticklabels(['← Left', '↓ Down', '→ Right', '↑ Up'])

os.makedirs("4x4", exist_ok=True)
plt.tight_layout()
plt.savefig("Q_table_Heatmap_4x4.png")
plt.close()

