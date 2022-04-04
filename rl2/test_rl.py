import numpy as np

import gym
import rl

def test_random_policy():
    pi = rl.random_policy(5, 3)
    print(pi)

    pi = rl.random_policy(5, 3, 0)
    print(pi)

    pi = rl.random_policy(5, 3, 0)
    print(pi)
    

def test_gym():
    pass


def test_play():
    env = gym.make('CartPole-v1')
    reward = rl.play(env, rl.Agent(env))
    print(f'reward : {reward}')


if __name__ == '__main__':
    test_random_policy()
    test_gym()
    test_play()
