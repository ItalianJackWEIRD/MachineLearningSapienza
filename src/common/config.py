Q_LEARNING_CONFIG = {
    'episodes': 1000,
    'learning_rate': 0.1,
    'gamma': 0.99,
    'epsilon_start': 1.0,
    'epsilon_min': 0.01,
    'epsilon_decay': 0.995,
    'buckets': (6, 12),
}

DQN_CONFIG = {
    'episodes': 500,
    'batch_size': 64,
    'gamma': 0.99,
    'epsilon_start': 1.0,
    'epsilon_min': 0.01,
    'epsilon_decay': 0.995,
    'learning_rate': 1e-3,
    'target_update': 10,
    'memory_size': 10000,
}
