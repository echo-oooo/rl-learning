import unittest

import gym
import numpy as np

from rl2 import mdp, rl


class TestMdp(unittest.TestCase):
    """ 测试 MDP 模块. """

    def test_env_to_mat(self):
        env = gym.make('FrozenLake-v1')
        env = env.unwrapped
        p, r = mdp.env_to_mat(env.P)
        p2, r2 = mdp.env_to_mat_2(env.P)
        self.assertTrue(np.all(p == p2) and np.all(r == r2))

        env = gym.make('CliffWalking-v1')
        env = env.unwrapped
        p, r = mdp.env_to_mat(env.P)
        p2, r2 = mdp.env_to_mat_2(env.P)
        self.assertTrue(np.all(p == p2) and np.all(r == r2))

    def t
