import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_rewards(rewards, title='Rewards Over Episodes'):
    df = pd.DataFrame({'Episode': range(len(rewards)), 'Reward': rewards})
    sns.lineplot(x='Episode', y='Reward', data=df)
    plt.title(title)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_')}.png")  # Save instead of show
    plt.close()
    
def smooth_rewards(rewards, weight=0.9):
    smoothed = []
    last = rewards[0]
    for r in rewards:
        smoothed_val = last * weight + (1 - weight) * r
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

def plot_smoothed_rewards(rewards, title='Smoothed Rewards'):
    smoothed = smooth_rewards(rewards)
    df = pd.DataFrame({'Episode': range(len(rewards)), 'Smoothed Reward': smoothed})
    sns.lineplot(x='Episode', y='Smoothed Reward', data=df)
    plt.title(title)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_')}.png")
    plt.close()

