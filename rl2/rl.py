"""
RL 常用函数.
"""

import numpy as np

class Agent:
    def __init__(self, env):
        self.env = env

    def decide(self, *args):
        return self.env.action_space.sample()

    def learn(self, *args):
        pass


def choose_action(pi, s):
    """ 选择动作. """
    pa = pi[s]
    a = np.random.choice(len(pa), p=pa)
    return a

def env_n(env):
    num_s = env.observation_space.n
    num_a = env.action_space.n
    return num_s, num_a


def play(env, agent, render=True, max_step=0):
    """ 进行一局游戏.

    :param env:
    :param agent:
    :param render:
    :param max_step:
    :return: 
    """
    done, total_reward, step = False, 0.0, 0

    def _render():
        if render:
            env.render()

    def _is_over() -> bool:
        nonlocal step
        step += 1
        return done or (max_step > 0 and step > max_step)

    s = env.reset()
    _render()
    while not _is_over():
        a = agent.decide(s)
        next_s, r, done, _ = env.step(a)
        _render()
        total_reward += r
        s = next_s
    return total_reward


def random_policy(num_s, num_a, seed=None):
    """ 生成随机策略. 
    
    :param num_s: 状态数量.
    :param num_a: 动作数量.    
    :return: 随机策略矩阵. 
    """
    if seed is None:
        pi = np.random.rand(num_s, num_a)
        for si in range(num_s):
            pi[si] = pi[si] / np.sum(pi[si])
        return pi
    else:
        np.random.seed(seed)
        pi = np.zeros((num_s, num_a))
        a = [np.random.randint(num_a) for _ in range(num_s)]
        for s in range(num_s):
            pi[s, a[s]] = 1.0
        return pi
        
        
    

