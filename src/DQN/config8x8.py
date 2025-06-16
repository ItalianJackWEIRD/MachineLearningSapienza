CONFIG = {
    "env_name": "FrozenLake-v1",
    "map_name": "8x8",
    "is_slippery": False,
    
    "hidden_layer_sizes": [128, 64],  # Two hidden layers
    
    "learning_rate": 0.001,
    "gamma": 0.9,   # Discount factor for future rewards
    "sync_rate": 1000, # number of steps after which the target network is updated with the policy network weights
    "memory_size": 80000,   # Size of the replay memory
    "mini_batch_size": 256, # Replay memory sample size for training
    
    "epsilon_start": 1.0,
    "epsilon_min": 0.01,
    "epsilon_decay": 1 / 10000,  # For example, decay epsilon over all episodes
    
    "episodes": 10000
}


# Note: "gamma" is the discount factor, The discount factor determines how much the agent values future rewards compared to immediate rewards.
# A value of γ = 0 makes the agent short-sighted: it only cares about immediate rewards.
# A value of γ = 1 makes the agent far-sighted: it cares about long-term rewards just as much as immediate ones.
# A value of γ = 0.9 (as in your config) means: The agent gives 90% of the weight to rewards one step in the future, 81% to two steps, 72.9% to three steps, and so on.


