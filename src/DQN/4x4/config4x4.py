CONFIG = {
    'env_name': 'FrozenLake-v1',
    'map_name': '4x4',
    'is_slippery': False,
    
    'episodes': 100,
    'learning_rate': 0.001,
    'memory_size': 5000,
    'mini_batch_size': 64,
    'gamma': 0.9,
    'sync_rate': 100,
    'epsilon_start': 1.0,
    'epsilon_min': 0.05,
    'epsilon_decay': 0.99,
    
    # Neural network: one hidden layer with 64 nodes
    'hidden_layer_sizes': [64],
}
