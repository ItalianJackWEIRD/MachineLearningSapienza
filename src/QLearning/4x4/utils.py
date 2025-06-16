import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

def plot_rewards(rewards, title='Rewards Over Episodes', save_dir=None):
    df = pd.DataFrame({'Episode': range(len(rewards)), 'Reward': rewards})
    sns.lineplot(x='Episode', y='Reward', data=df)
    plt.title(title)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.tight_layout()
    # Directory di salvataggio
    if save_dir is None:
        save_dir = os.path.dirname(__file__)
    filename = os.path.join(save_dir, f"{title.replace(' ', '_')}.png")
    
    plt.savefig(filename)
    plt.close()
    
def smooth_rewards(rewards, weight=0.9):
    smoothed = []
    last = rewards[0]
    for r in rewards:
        smoothed_val = last * weight + (1 - weight) * r
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

def plot_smoothed_rewards(rewards, title='Smoothed Rewards', save_dir=None):
    smoothed = smooth_rewards(rewards)
    df = pd.DataFrame({'Episode': range(len(rewards)), 'Smoothed Reward': smoothed})
    sns.lineplot(x='Episode', y='Smoothed Reward', data=df)
    plt.title(title)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.tight_layout()
    # Directory di salvataggio
    if save_dir is None:
        save_dir = os.path.dirname(__file__)
    filename = os.path.join(save_dir, f"{title.replace(' ', '_')}.png")
    
    plt.savefig(filename)
    plt.close()

