"""
马尔可夫决策过程.

------------------------------------------------------------------------------

策略:   pi(a | s) :   [S x A] ∈ [0, 1]
动力:   p(s', r | s, a) : 
奖励:       r(s, a): [S x A] ∈ [0, 1]
折扣因子:   gamma ∈ [0, 1]

------------------------------------------------------------------------------

iterate_value: 价值迭代.
iterate_policy: 策略迭代.
evaluate_policy: 策略评估.
improve_policy: 策略改进.
q_to_pi: 动作价值函数生成策略.
v_to_q: 状态价值函数生成动作价值函数.
env_to_mat: 环境转换为概率矩阵.

"""

import numpy as np
import rl


class Mdp:
    def __init__(self, **kwargs):
        """ 初始化. 
        
        :param env: 根据 env 自动初始化策略.
        :param pi: 手动指定策略.
        """
        if 'env' in kwargs:
            env = kwargs['env']
            num_s, num_a = rl.env_n(env)
            self.policy = rl.random_policy(num_s, num_a)
        if 'pi' in kwargs:
            self.policy = kwargs['pi']

    def decide(self, s):
        return rl.choose_action(self.policy, s)

    def learn(self, env,  gamma=0.9, type_: str = 'policy', max_step=10):
        """ 使用策略迭代提升获取策略. 

        :param env: 环境.
        :param type_: 学习类型. 'value' 价值迭代方法; 'policy' 策略迭代方法.
        """
        p, r = env_to_mat(env.P)
        if type_ == 'value':
            self.policy, _ = iterate_value(p, r, gamma=gamma)
        else:
            self.policy, _ = iterate_policy(p, r, gamma=gamma)


def iterate_policy(p, r, gamma=0.9, max_step=25):
    """ 使用策略迭代提升获取策略. """
    pi_curr = rl.random_policy(*r.shape)
    pi_next = pi_curr.copy()
    for _ in range(max_step):
        v = evaluate_policy(p, r, pi_curr, gamma=gamma)
        pi_next = improve_policy(p, r, v, gamma=gamma)
        if np.all(pi_curr == pi_next):
            break
        pi_curr = pi_next.copy()
    return pi_next, evaluate_policy(p, r, pi_next, gamma=gamma)


def iterate_value(p, r, gamma=0.9, max_step=25, eps=1e-5):
    """ 使用价值迭代获取策略. """
    v, v_next = np.zeros((r.shape[0], )), np.zeros((r.shape[0], ))
    for _ in range(max_step):
        temp = r + gamma * np.dot(p, v)
        v_next = np.max(temp, axis=1)
        if np.max(np.abs(v_next - v)) < eps:
            break
        v = v_next.copy()
    q = v_to_q(p, r, v_next, gamma=gamma)
    pi = q_to_pi(q)
    return pi, v


def env_evaluate_policy(P, pi, gamma):
    """ 针对 gym 环境的策略评估. 

    :param P:   gym.env.P
    :param pi:  策略, pi(a|s).
    :param gamma: 折扣因子.
    :return: 状态价值函数, v(s).
    """
    p, r = env_to_mat(P)
    return evaluate_policy(p, r, pi, gamma)


def evaluate_policy(p, r, pi, gamma=0.9, eps=1e-5, max_round=25):
    """ 策略评估. 
    
    :param p: 状态转移矩阵. p(s'|s, a)
    :param r: 状态/动作奖励矩阵. r(s, a)
    :param pi: 策略, pi(a|s).
    :param gamma: 折扣因子.
    :return: 状态价值函数. v(s)
    """
    num_s = pi.shape[0]
    v, v_next = np.zeros((num_s, )), np.zeros((num_s, ))
    for _ in range(max_round):
        q = v_to_q(p, r, v, gamma=gamma)
        v_next = np.sum(np.multiply(pi, q), axis=1)
        delta = np.max(np.abs(v - v_next))
        v = v_next.copy()
        if delta < eps:
            break
    return v_next


def q_to_pi(q):
    """ 从动作价值函数生成策略. """
    pi = np.zeros_like(q)
    for s in range(q.shape[0]):
        pi[s] = (q[s] == np.max(q[s]))
        pi[s] = pi[s] / np.sum(pi[s])
    return pi


def improve_policy(p, r, v, gamma=0.9):
    """ 提升策略.
    """
    q = v_to_q(p, r, v, gamma=gamma)
    pi = np.zeros_like(q)

    num_s = q.shape[0]
    for s in range(num_s):
        pi[s] = (q[s] == np.max(q[s]))
        pi[s] = pi[s] / np.sum(pi[s])
    return pi


def v_to_q(p, r, v, sa=None, gamma=0.9):
    """ 状态价值转换为动作价值.
   
    q(s,a) = r(s,a)+ gamma * sum( p(s'|s,a) * v(s') )

    :param p: 状态转移矩阵. p(s'|s, a)
    :param r: 状态/动作奖励矩阵. r(s, a)
    :param v: 状态价值函数. v(s)
    :param sa:  (s,a) 坐标. None 表示计算整个 q 矩阵.
    :param gamma: 折扣因子.
    :return: 动作价值函数. q(s, a)
    """
    if sa is None:
        q = r + gamma * np.dot(p, v)
        return q
    else:
        s, a = sa
        q = r[s, a] + gamma * np.dot(p[s, a, :], v)
        return q


def env_to_mat(P):
    """ 将环境变量转换为矩阵.

    :param P: gym.env.P
    :return: (p, r)
    """
    p = _env_to_p(P)
    r = _env_to_r(P)
    return p, r


def _env_to_r(P):
    """ 从环境动力 P 转换为 “状态-动作”期望奖励矩阵 R(S, A). 
    """
    num_s, num_a = _env_n(P)
    ret = np.zeros((num_s, num_a))
    for s in range(num_s):
        for a in range(num_a):
            ret[s, a] = _env_r(P, s, a)
    return ret


def _env_to_p(P):
    num_s, num_a = _env_n(P)
    ret = np.zeros((num_s, num_a, num_s))
    for s in range(num_s):
        for a in range(num_a):
            for s_next in range(num_s):
                ret[s, a, s_next] = _env_p(P, s, a, s_next)
    return ret


def _env_p(P, s, a, s_next):
    """ 计算状态转移概率 p(s'|s,a) """
    ret = 0
    for val in P[s][a]:
        ret += val[0] if val[1] == s_next else 0.0
    return ret


def _env_n(P):
    """ 动力的( 状态数, 动作数 ). """
    return len(P), len(P[0])


def _env_r(P, s, a):
    """ 计算 r(s, a) 
    :param P: env.unwrapped.P [S][A] -> (p, next_s, r, done)
    """
    ret = 0
    for val in P[s][a]:
        ret += val[0] * val[2]
    return ret
