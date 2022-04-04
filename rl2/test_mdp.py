import unittest
import gym
import numpy as np

import rl
import mdp


def test_mdp_policy_frozenlake():
    env = gym.make('FrozenLake-v1')
    env = env.unwrapped

    agent = mdp.Mdp(env=env)
    v0 = [rl.play(env, agent, render=False) for _ in range(100)]
    print(f'before learn : {np.mean(v0)}')

    agent.learn(env, type_='policy')
    v1 = [rl.play(env, agent, render=False) for _ in range(100)]
    print(f'after learn : {np.mean(v1)}')

    # rl.play(env, agent)


def test_mdp_value_frozenlake():
    env = gym.make('FrozenLake-v1')
    env = env.unwrapped

    agent = mdp.Mdp(env=env)
    v0 = [rl.play(env, agent, render=False) for _ in range(100)]
    print(f'before learn : {np.mean(v0)}')

    agent.learn(env, type_='value')
    v1 = [rl.play(env, agent, render=False) for _ in range(100)]
    print(f'after learn : {np.mean(v1)}')

    # rl.play(env, agent)


def test_mdp_value_cliffwalking():
    env = gym.make('CliffWalking-v0')
    env = env.unwrapped

    p, r = mdp.env_to_mat(env.P)
    pi = rl.random_policy(*r.shape, type_='avg')
    v = mdp.evaluate_policy(p, r, pi)
    
    agent = mdp.Mdp(env=env)
    agent.learn(env, type_='policy')
    rl.play(env, agent)


def test_util():
    """ 测试工具函数. """
    # env = gym.make('CliffWalking-v0')
    env = gym.make('FrozenLake-v1')
    env = env.unwrapped
    p, r = mdp.env_to_mat(env.P)

    p2, r2 = mdp.env_to_mat_2(env.P)

    print(f'{np.all(p == p2)}')
    print(f'{np.all(r == r2)}')
    
    
    
    print(f'{len(env.P[0])}')
    print(f'{r}')

    # 测试 v2q
    v = np.random.random((p.shape[0], ))
    v_m = mdp.v_to_q(p, r, v)
    v_00 = mdp.v_to_q(p, r, v, (0, 0))
    v_33 = mdp.v_to_q(p, r, v, (3, 3))
    print(f'v2q() passed : {(v_m[0,0] == v_00) and (v_m[3,3] == v_33)}')

    # 测试 evaluate_policy
    # pi = rl.random_policy(*rl.env_n(env), type_='avg')
    # v = mdp.evaluate_policy(p, r, pi)
    # print(f'v(s) @ pi(1) : {v}')

    # pi2 = rl.random_policy(*rl.env_n(env), seed=2)
    # v2 = mdp.evaluate_policy(p, r, pi2)
    # # print(f'v(s) @ pi(2) : {v2}')

    # 测试价值迭代.
    pi1, v1 = mdp.iterate_value(p, r, gamma=1)

    # 测试策略迭代.
    pi2, v2 = mdp.iterate_policy(p, r, gamma=1)

    print(f'iterate pi diff : {pi1 - pi2}')
    print(f'iterate value max-diff: {np.max(np.abs(v1 - v2))} ')

    pass


if __name__ == '__main__':
    test_util()
    # test_mdp_value_frozenlake()
    # test_mdp_policy_frozenlake()
    # test_mdp_value_cliffwalking()
