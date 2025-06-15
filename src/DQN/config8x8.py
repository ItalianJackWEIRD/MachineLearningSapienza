CONFIG = {
    "env_name": "FrozenLake-v1",
    "map_name": "8x8",
    "is_slippery": False,
    
    "hidden_layer_sizes": [128, 64],  # Two hidden layers
    
    "learning_rate": 0.001,
    "memory_size": 80000,
    "mini_batch_size": 256,
    "gamma": 0.9,
    "sync_rate": 1000,
    
    "epsilon_start": 1.0,
    "epsilon_min": 0.01,
    "epsilon_decay": 1 / 10000,  # For example, decay epsilon over all episodes
    
    "episodes": 10000
}
