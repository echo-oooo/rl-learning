import unittest

import gym
import numpy as np

from rl2 import rl


class TestRl(unittest.TestCase):
    """ 测试 rl 模块. """

    def test_rl(self):
        env = gym.make('FrozenLake')
        num_s, num_a = rl.env_n(env)
        self.assertTrue(num_s > 0)

    def test_random_policy(self):
        pi = rl.random_policy(5, 3)
        pis = np.sum(pi, axis=1)
        np.testing.assert_almost_equal(pis, 1)

        pi = rl.random_policy(5, 3, seed=0)
        pi2 = rl.random_policy(5, 3, seed=0)
        self.assertTrue(np.all(pi == pi2))

        pi = rl.random_policy(5, 3, type_='avg')
        np.testing.assert_almost_equal(pi, 1.0/3)

    def test_env_n(self):
        env = gym.make('FrozenLake')
        s, a = rl.env_n(env)
        self.assertTrue(s == 16 and a == 4)

    def test_play(self):
        env = gym.make('FrozenLake')
        agent = rl.Agent(env)
        r = rl.play(env, agent, render=False)
        self.assertTrue(r is not None)
