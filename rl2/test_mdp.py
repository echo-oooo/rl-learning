import gym
import numpy as np

import rl
import mdp

def test_mdp():
    env = gym.make('FrozenLake-v1')
    env = env.unwrapped

    agent = mdp.Mdp(env=env)
    v0 = [rl.play(env, agent, render=False) for _ in range(100)]
    print(f'before learn : {np.mean(v0)}')

    agent.learn(env)
    v1 = [rl.play(env, agent, render=False) for _ in range(100)]
    print(f'after learn : {np.mean(v1)}')


def test_util():
    a = np.ones((5, 3))
    b = np.ones((5, 3))
    env = gym.make('FrozenLake-v1')
    env = env.unwrapped
    p, r = mdp.env_to_mat(env.P)


    pi = rl.random_policy(*rl.env_n(env))
    v = mdp.evaluate_policy(p, r, pi)
    print(f'v(s) : {v}')
    
    pi2 = rl.random_policy(*rl.env_n(env))
    v2 = mdp.evaluate_policy(p, r, pi2)
    print(f'v(s) : {v2}')

    pi = mdp.iterate_policy(p, r)
    pass
    

if __name__ == '__main__':
    test_util()
    test_mdp()

