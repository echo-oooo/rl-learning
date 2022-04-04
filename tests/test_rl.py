import unittest

import gym
from rl2 import rl


class TestRl(unittest.TestCase):
    """ 测试 rl 模块. """
    def test_rl(self):
        env = gym.make('FrozenLake')
        num_s, num_a = rl.env_n(env)
        self.assertTrue(num_s > 0)
