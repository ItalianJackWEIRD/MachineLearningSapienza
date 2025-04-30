import argparse
import gymnasium as gym
from q_learning.q_learning import train_q_learning
from dqn.dqn import train_dqn
from common.config import Q_LEARNING_CONFIG, DQN_CONFIG

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', choices=['q_learning', 'dqn'], required=True)
    args = parser.parse_args()

    env = gym.make('CartPole-v1')

    if args.algo == 'q_learning':
        train_q_learning(env, Q_LEARNING_CONFIG)
    else:
        train_dqn(env, DQN_CONFIG)

    env.close()

if __name__ == '__main__':
    main()
