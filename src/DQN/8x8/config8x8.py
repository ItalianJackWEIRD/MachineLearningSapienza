CONFIG = {
    "env_name": "FrozenLake-v1",
    "map_name": "8x8",
    "is_slippery": False,

    # Architettura
    "hidden_layer_sizes": [128, 64],

    # Ottimizzazione
    "learning_rate": 0.001,      # un po’ più conservativo per stabilità
    "gamma": 0.99,               # reward futuri quasi totalmente considerati
    "sync_rate": 200,            # sincronizzo target meno frequentemente per stabilità

    # Experience replay
    "memory_size": 20000,        # più esperienza per copertura dello spazio
    "mini_batch_size": 64,       # batch più grandi per gradienti più stabili

    # Epsilon-greedy
    "epsilon_start": 1.0,
    "epsilon_min": 0.01,
    "epsilon_decay": (1.0 - 0.01) / 1500,  
    # scalo da 1→0.01 su ~1500 episodi, gli ultimi 500 rimangono a ε≈0.01 per consolidare

    # Training
    "episodes": 2000
}
