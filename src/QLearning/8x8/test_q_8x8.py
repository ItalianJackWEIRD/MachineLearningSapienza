import numpy as np
import gymnasium as gym
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from q_learning import QLearningAgent
from config_q_8x8 import CONFIG
import os

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

q_table_path = "8x8/q_table_custom_8x8.npy"
if not os.path.exists(q_table_path):
    raise FileNotFoundError(f"Q-table not found at {q_table_path}. Please run training first.")
q_table = np.load(q_table_path)
env = gym.make(CONFIG['env_name'], desc=custom_map_8x8, is_slippery=False)
state_size = env.observation_space.n
action_size = env.action_space.n
goal_state = state_size - 1

agent = QLearningAgent(state_size, action_size, CONFIG)
agent.epsilon = 0.0
agent.epsilon_decay = 1.0
agent.epsilon_min = 0.0
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

    print(f"[Test Ep {ep + 1}] Total reward: {total_reward:.3f}")
    rewards.append(total_reward)
    actions_per_episode.append(episode_actions)
    if total_reward > 0:
        successes += 1

print(f"\nTest results over {n_episodes} episodes:")
print(f"Success rate: {successes / n_episodes:.2%}")
print(f"Average reward: {np.mean(rewards):.3f}")

os.makedirs("8x8", exist_ok=True)
with open("8x8/test_results_8x8.txt", "w") as f:
    f.write(f"Success rate: {successes / n_episodes:.2%}\n")
    f.write(f"Average reward: {np.mean(rewards):.3f}\n")

plt.figure()
sns.histplot(rewards, bins=[0, 0.5, 1], discrete=True)
plt.title("Test Reward Distribution (8x8)")
plt.xlabel("Reward")
plt.ylabel("Frequency")
plt.xticks([0, 1])
plt.tight_layout()
plt.savefig("8x8/Test_Reward_Distribution_8x8.png")
plt.close()

best_actions = np.argmax(agent.q_table, axis=1)
plt.figure()
plt.bar(range(len(best_actions)), best_actions)
plt.title("Best Action per State (8x8)")
plt.xlabel("State")
plt.ylabel("Best Action")
plt.tight_layout()
plt.savefig("8x8/Test_Best_Action_Per_State_8x8.png")
plt.close()
