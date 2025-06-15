import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from frozenlake_dqn_model_8x8 import DQN, ReplayMemory
import tensorflow as tf
import random
from config8x8 import CONFIG

ACTIONS = ['L', 'D', 'R', 'U']  # for printing 0,1,2,3 -> 'Left', 'Down', 'Right', 'Up'


'''
Converts a state (int) to a tensor representation.
For example, the FrozenLake 4x4 map has 4x4=16 states numbered from 0 to 15. 

Parameters: state=1, num_states=16
Return: tensor([0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
'''
def state_to_input(state, num_states):
    one_hot = np.zeros(num_states)
    one_hot[state] = 1
    return np.expand_dims(one_hot, axis=0)

def train():
    env = gym.make(CONFIG['env_name'], map_name=CONFIG['map_name'], is_slippery=CONFIG['is_slippery'])
    num_states = env.observation_space.n
    num_actions = env.action_space.n

    # Get hidden layers from config (assumes list like [128, 64])
    hidden_layers = CONFIG.get('hidden_layer_sizes', [128, 64])
    
    # Instantiate DQN with *unpack hidden layers*
    policy_dqn = DQN(num_states, *hidden_layers, num_actions)       # STEP 1 : Create the policy network
    target_dqn = DQN(num_states, *hidden_layers, num_actions)       # STEP 2 : Create the target network identical to the policy network
    target_dqn.set_weights(policy_dqn.get_weights())

    optimizer = tf.keras.optimizers.Adam(learning_rate=CONFIG['learning_rate'])     # NN optimizer
    loss_fn = tf.keras.losses.MeanSquaredError()    # NN loss function 

    memory = ReplayMemory(CONFIG['memory_size'])
    mini_batch_size = CONFIG['mini_batch_size']
    gamma = CONFIG['gamma']
    sync_rate = CONFIG['sync_rate']
    epsilon = CONFIG['epsilon_start']
    epsilon_min = CONFIG.get('epsilon_min', 0.0)
    epsilon_decay = CONFIG.get('epsilon_decay', 1 / CONFIG['episodes'])
    episodes = CONFIG['episodes']

    rewards_per_episode = np.zeros(episodes)        # keep track of rewards per episode
    epsilon_history = []    # keep track of epsilon decay
    step_count = 0      # step count for syncing policy -> target network

    for i in range(episodes):
        state = env.reset()[0]
        done = False

        while not done:
            if random.random() < epsilon:   # STEP 3 : Epsilon-greedy algorithm    
                action = env.action_space.sample()
            else:
                q_values = policy_dqn(state_to_input(state, num_states))
                action = tf.argmax(q_values[0]).numpy()

            new_state, reward, terminated, truncated, _ = env.step(action)  # Take action in the environment

            done = terminated or truncated
            memory.append((state, action, new_state, reward, done)) # Store transition in memory
            state = new_state   # Update state for next step
            step_count += 1

        if reward == 1:
            rewards_per_episode[i] = 1

        # Check memory to see if we have enough traning data and if the episode was successful
        if len(memory) >= mini_batch_size and np.sum(rewards_per_episode) > 0:
            mini_batch = memory.sample(mini_batch_size)
            states, actions, new_states, rewards, dones = zip(*mini_batch)

            states = np.vstack([state_to_input(s, num_states) for s in states])
            new_states = np.vstack([state_to_input(s, num_states) for s in new_states])
            actions = np.array(actions)
            rewards = np.array(rewards)
            dones = np.array(dones)

            future_q = target_dqn.predict(new_states).max(axis=1)
            targets = policy_dqn.predict(states)
            for idx, action in enumerate(actions):
                targets[idx, action] = rewards[idx] if dones[idx] else rewards[idx] + gamma * future_q[idx]

            policy_dqn.compile(optimizer=optimizer, loss=loss_fn)
            policy_dqn.fit(states, targets, verbose=0)

            # Decay epsilon but keep above min value
            epsilon = max(epsilon - epsilon_decay, epsilon_min)
            epsilon_history.append(epsilon)

            # STEP 10 : Sync target network with policy network weights after a certain number of steps   
            if step_count > sync_rate:
                target_dqn.set_weights(policy_dqn.get_weights())
                step_count = 0

        if (i+1) % 1000 == 0:
            print(f"Episode {i+1}/{episodes} - Epsilon: {epsilon:.3f} - Recent avg reward: {np.mean(rewards_per_episode[max(0,i-999):i+1]):.3f}")

    policy_dqn.save_weights("frozen_lake_dql_8x8.weights.h5")

    plt.figure(figsize=(12, 5))

    window_size = 100
    cumulative_rewards = [np.sum(rewards_per_episode[max(0, i - window_size + 1):(i + 1)]) for i in range(len(rewards_per_episode))]

    plt.subplot(1, 2, 1)
    plt.plot(cumulative_rewards, label='Cumulative Reward (100 eps)')
    plt.title('Training Performance on FrozenLake 8x8')
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward (Last 100 Episodes)')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epsilon_history, label='Epsilon (Exploration Rate)', color='orange')
    plt.title('Epsilon Decay over Training')
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("frozen_lake_dql_8x8.png")
    plt.show()

if __name__ == '__main__':
    train()
