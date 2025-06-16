import numpy as np
import gymnasium as gym
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from q_learning import QLearningAgent
from config_q_4x4 import CONFIG

q_table = np.load("q_table_4x4.npy")

env = gym.make(
    CONFIG['env_name'],
    map_name=CONFIG['map_name'],
    is_slippery=CONFIG['is_slippery']
)
state_size = env.observation_space.n
action_size = env.action_space.n

agent = QLearningAgent(state_size, action_size, epsilon=0.0)
agent.q_table = q_table

n_episodes = CONFIG['test_episodes']
successes = 0
rewards = []
actions_per_episode = []
for ep in range(n_episodes):
    state, _ = env.reset()
    done = False
    total_reward = 0
    episode_actions = []
    while not done:
        action = agent.choose_action(state)
        next_state, reward, done, truncated, _ = env.step(action)
        episode_actions.append(action)
        state = next_state
        total_reward += reward
    rewards.append(total_reward)
    actions_per_episode.append(episode_actions)
    if total_reward > 0:
        successes += 1

print(f"\nTest results over {n_episodes} episodes:")
print(f"Success rate: {successes / n_episodes:.2%}")
print(f"Average reward: {np.mean(rewards):.3f}")

with open("test_results_4x4.txt", "w") as f:
    f.write(f"Success rate: {successes / n_episodes:.2%}\n")
    f.write(f"Average reward: {np.mean(rewards):.3f}\n")

plt.figure()
sns.histplot(rewards, bins=[0, 0.5, 1], discrete=True)
plt.title("Test Reward Distribution (4x4)")
plt.xlabel("Reward")
plt.ylabel("Frequency")
plt.xticks([0, 1])
plt.tight_layout()
plt.savefig("Test_Reward_Distribution_4x4.png")
plt.close()

best_actions = np.argmax(agent.q_table, axis=1)
plt.figure(figsize=(10, 6))
plt.bar(range(len(best_actions)), best_actions, color='skyblue')
plt.title("Best Action per State from Trained Policy (4x4)")
plt.xlabel("State (0 to 63)")
plt.ylabel("Best Action (0=Left,1=Down,2=Right,3=Up)")
plt.xticks(ticks=range(0, len(best_actions), 4))
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("Test_Best_Action_Per_State_4x4.png")
plt.close()
