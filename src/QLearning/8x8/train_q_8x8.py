import gymnasium as gym
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from q_learning import QLearningAgent
from config_q_8x8 import CONFIG
import os
import time

custom_map_8x8 = [
    "SFFFFFFH",
    "FFFFFHFF",
    "FFFFHFHF",
    "FFFFFHFF",
    "FFFFFHFF",
    "HFFFFHFH",
    "FHFFFHFF",
    "FFHFFFFG"
]

desc = np.array([list(row) for row in custom_map_8x8])
map_size = desc.shape[0]
env = gym.make(CONFIG['env_name'], desc=custom_map_8x8, is_slippery=CONFIG['is_slippery'])
agent = QLearningAgent(
    state_size=env.observation_space.n,
    action_size=env.action_space.n,
    config=CONFIG
)
episodes = CONFIG['episodes']
rewards = []
epsilons = []

os.makedirs("8x8", exist_ok=True)

for ep in range(episodes):
    ep_start_time = time.time()
    state, _ = env.reset()
    total_reward = 0
    done = False
    steps = 0

    while not done:
        action = agent.choose_action(state)
        next_state, reward, done, truncated, _ = env.step(action)
        agent.update(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        steps += 1

    rewards.append(total_reward)
    epsilons.append(agent.epsilon)

    ep_duration = time.time() - ep_start_time

total_time = time.time() - start_time
print(f"\n🏁 Training completato in {total_time:.2f} secondi ({total_time/60:.2f} minuti)")

df_rewards = pd.DataFrame({'Episode': range(len(rewards)), 'Reward': rewards})
sns.lineplot(x='Episode', y='Reward', data=df_rewards)
plt.title('Tabular Q-Learning on Custom FrozenLake 8x8')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.tight_layout()
plt.savefig("8x8/Tabular_Q-Learning_on_Custom_FrozenLake_8x8.png")
plt.close()

successes = [int(r > 0) for r in rewards]
rolling_success = pd.Series(successes).rolling(window=100).mean()

sns.lineplot(x=range(len(rolling_success)), y=rolling_success)
plt.title("Rolling Success Rate (window=100)")
plt.xlabel("Episode")
plt.ylabel("Success Rate")
plt.tight_layout()
plt.savefig("8x8/Rolling_Success_Rate_Custom_8x8.png")
plt.close()

plt.figure()
plt.plot(range(len(epsilons)), epsilons)
plt.title("Epsilon Decay Over Episodes")
plt.xlabel("Episode")
plt.ylabel("Epsilon (Exploration Rate)")
plt.tight_layout()
plt.savefig("8x8/Epsilon_Decay_Custom_8x8.png")
plt.close()

np.save("8x8/q_table_custom_8x8.npy", agent.q_table)

color_map = {'S': 'green', 'F': 'lightblue', 'H': 'black', 'G': 'red'}

fig, ax = plt.subplots(figsize=(7, 7))
for row in range(map_size):
    for col in range(map_size):
        tile = desc[row][col]
        ax.add_patch(plt.Rectangle((col, map_size - row - 1), 1, 1, color=color_map[tile], ec='gray'))
        ax.text(col + 0.5, map_size - row - 1 + 0.5, tile, ha='center', va='center',
                color='white' if tile == 'H' else 'black', fontsize=12)

ax.set_xlim(0, map_size)
ax.set_ylim(0, map_size)
ax.set_xticks(range(map_size + 1))
ax.set_yticks(range(map_size + 1))
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_title('FrozenLake Map Layout (Custom 8x8)')
plt.grid(True)
plt.gca().set_aspect('equal', adjustable='box')
plt.tight_layout()
plt.savefig("8x8/FrozenLake_Custom_8x8_Layout.png")
plt.close()

plt.figure(figsize=(12, 10))
ax = sns.heatmap(agent.q_table, annot=False, cmap='viridis', cbar_kws={'label': 'Q-value'})
ax.set_title("Q-Table Heatmap (FrozenLake Custom 8x8)", fontsize=14)
ax.set_xlabel("Actions", fontsize=12)
ax.set_ylabel("States", fontsize=12)
ax.set_xticklabels(['← Left', '↓ Down', '→ Right', '↑ Up'])
plt.tight_layout()
plt.savefig("8x8/Q_table_Heatmap_Custom_8x8.png")
plt.close()
