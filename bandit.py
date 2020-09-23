# pylint: disable=no-member, invalid-name, not-callable, missing-docstring
import itertools
import torch


def inv_action(s):
    return s.replace('A', 'b').replace('B', 'A').replace('b', 'B')


def inv_reward(s):
    return s.replace('1', 'o').replace('0', '1').replace('o', '0')


def str_prod(*iterators, n=1):
    return tuple([''.join(x) for x in itertools.product(*iterators, repeat=n)])


def init(n_arms, mem):
    arms = tuple('ABCDEFGH'[:n_arms])
    actions = str_prod(arms, '01')
    states = str_prod('+-', str_prod('01', n=mem))
    rewards = torch.tensor([1.0 if s[0] == '+' else -1.0 for s in states])
    return states, actions, arms, rewards


def master_matrix(states, actions, prob):
    """
    compute the matrix M_{ss,s,a} = P[ss | s, a]
    """
    mm = torch.tensor([
        [
            [
                prob(ss, s, a)
                for a in actions
            ]
            for s in states
        ]
        for ss in states
    ], dtype=torch.float64)
    return mm


def transfer_matrix(pi, mm):
    return torch.einsum('ja,ija->ij', pi, mm)


def steadystate(m, eps=1e-6):
    p = m.new_ones(len(m)) / len(m)

    for _ in itertools.count():
        m = m @ m

        for _ in range(2):
            p_ = m @ p

            d = (p_ - p).abs().max().item()
            p = p_

            if d < eps:
                break
        if d < eps:
            break

    return p


def uniform_grid(n):
    a = 1 / (2 + 2 * n)
    f = torch.linspace(a, 1.0 - a, n)
    fs = torch.stack(torch.meshgrid(f, f), dim=-1).reshape(-1, 2)
    return fs


def avg_gain(rewards, mms, reset, eps, pi):
    r = pi.new_ones(len(pi), len(pi)) / len(pi)  # reset transfer matrix

    g = 0
    for mm in mms:
        m = transfer_matrix(pi, mm)
        m = (1 - reset) * m + reset * r
        p = steadystate(m, eps)
        g += torch.dot(rewards, p)
    g = g / len(mms)
    return g


def grad_fn(rewards, mms, reset, eps, w):
    w = w.clone()
    w.requires_grad_()
    g = avg_gain(rewards, mms, reset, eps, w.softmax(1))
    g.backward()
    return w.grad.clone(), g.item()
