# pylint: disable=no-member, invalid-name, not-callable, missing-docstring
import itertools
import torch


def inv_action(s):
    return s.replace('A', '__b__').replace('B', 'A').replace('__b__', 'B')


def inv_reward(s):
    return s.replace('1', '__z__').replace('0', '1').replace('__z__', '0')


def str_prod(*iterators, n=1):
    return tuple([''.join(x) for x in itertools.product(*iterators, repeat=n)])


def init(n_arms, mem, mem_type):
    arms = tuple('ABCDEFGH'[:n_arms])

    if mem_type == 'shift':
        actions = str_prod(arms, '01')
        states = str_prod('+-', str_prod('01', n=mem))
        rewards = torch.tensor([1.0 if s[0] == '+' else -1.0 for s in states])
        def prob(f, ss, s, a):
            # s = +1101
            # a = A0
            #ss = -1010
            if ss[1:] == s[2:] + a[1]:
                fa = f[arms.index(a[0])]
                return fa if ss[0] == '+' else 1 - fa
            return 0

    elif mem_type == 'ram':
        actions = str_prod(arms, str_prod('01', n=mem))
        states = str_prod('+-', str_prod('01', n=mem))
        rewards = torch.tensor([1.0 if s[0] == '+' else -1.0 for s in states])
        def prob(f, ss, _s, a):
            # s = +1101
            # a = A0110
            #ss = -0110
            if ss[1:] == a[1:]:
                fa = f[arms.index(a[0])]
                return fa if ss[0] == '+' else 1 - fa
            return 0

    elif mem_type == 'actions':
        actions = arms
        states = str_prod(str_prod(arms, n=mem), str_prod('+-', n=mem))
        rewards = torch.tensor([1.0 if s[-1] == '+' else -1.0 for s in states])
        def prob(f, ss, s, a):
            # s = AAB++-
            # a = B
            #ss = ABB+--
            if ss[:mem] == s[1:mem] + a and ss[mem:-1] == s[mem + 1:]:
                fa = f[arms.index(a)]
                return fa if ss[-1] == '+' else 1 - fa
            return 0

    elif mem_type in ['linear', 'restless']:
        if mem_type == 'linear':
            actions = str_prod(arms, '<.>')
        if mem_type == 'restless':
            actions = str_prod(arms, '<>')

        states = str_prod('+-', arms, [f"{i:04d}" for i in range(mem)])
        rewards = torch.tensor([1.0 if s[0] == '+' else -1.0 for s in states])
        def prob(f, ss, s, a):
            # s = +B0120
            # a = A>
            #ss = +A0121
            if (a[1] == '.' and s[2:] == ss[2:]) or (a[1] == '>' and min(int(s[2:]) + 1, mem - 1) == int(ss[2:])) or (a[1] == '<' and max(int(s[2:]) - 1, 0) == int(ss[2:])):
                if ss[1] == a[0]:
                    fa = f[arms.index(a[0])]
                    return fa if ss[0] == '+' else 1 - fa
            return 0

    return states, actions, arms, rewards, prob


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


def grad_fn(states, rewards, mms, reset, eps, ab_sym, w):
    w = w.clone()
    w.requires_grad_()
    if ab_sym:
        pi = w.softmax(1)
        pi = torch.stack([
            pi[i] if i < len(pi) else pi[states.index(inv_action(s))].flip(0)
            for i, s in enumerate(states)
        ])
    else:
        pi = w.softmax(1)
    g = avg_gain(rewards, mms, reset, eps, pi)
    g.backward()
    return w.grad.clone(), g.item()
