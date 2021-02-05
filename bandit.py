# pylint: disable=no-member, invalid-name, not-callable, missing-docstring
import itertools
import torch
import collections


def inv_action(s):
    return s.replace('A', '__b__').replace('B', 'A').replace('__b__', 'B')


def inv_reward(s):
    return s.replace('1', '__z__').replace('0', '1').replace('__z__', '0')


def str_prod(*iterators, n=1):
    return tuple([''.join(x) for x in itertools.product(*iterators, repeat=n)])


def init(n_arms, mem, mem_type):
    arms = tuple('ABCDEFGH'[:n_arms])

    n_init_states = 0

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
        states = []
        for n in range(mem + 1):
            states += str_prod([" " * (mem - n)], str_prod(arms, n=n), [" " * (mem - n)], str_prod("+-", n=n))
        n_init_states = 1
        rewards = torch.tensor([{'+': +1.0, '-': -1.0, ' ': 0.0}[s[-1]] for s in states])
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

        states = str_prod(('0 ',) + str_prod('+-', arms), [f"{i:02d}" for i in range(mem)])
        n_init_states = mem
        rewards = torch.tensor([{'+': 1.0, '-': -1.0, '0': 0}[s[0]] for s in states])
        def prob(f, ss, s, a):
            # s = +B0120
            # a = A>
            #ss = +A0121
            if (a[1] == '.' and s[2:] == ss[2:]) or (a[1] == '>' and min(int(s[2:]) + 1, mem - 1) == int(ss[2:])) or (a[1] == '<' and max(int(s[2:]) - 1, 0) == int(ss[2:])):
                if ss[1] == a[0]:
                    fa = f[arms.index(a[0])]
                    return fa if ss[0] == '+' else 1 - fa
            return 0

    if n_init_states == 0:
        n_init_states = len(states)

    return states, actions, arms, rewards, prob, n_init_states


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


def transfer_matrix(pi, mm, reset, p0):
    r = pi.new_zeros(len(pi), len(pi))
    r[:len(p0), :] = p0[:, None]

    m = torch.einsum('ja,ija->ij', pi, mm)
    m = (1 - reset) * m + reset * r
    return m


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


def avg_gain(rewards, mms, reset, eps, pi, p0):
    g = 0
    for mm in mms:
        m = transfer_matrix(pi, mm, reset, p0)
        p = steadystate(m, eps)
        g += torch.dot(rewards, p)
    g = g / len(mms)
    return g


def grad_fn(states, rewards, mms, reset, eps, ab_sym, w_pi, w_p0):
    w_pi = w_pi.clone()
    w_pi.requires_grad_()

    if ab_sym:
        pi = w_pi.softmax(1)
        pi = torch.stack([
            pi[i] if i < len(pi) else pi[states.index(inv_action(s))].flip(0)
            for i, s in enumerate(states)
        ])
    else:
        pi = w_pi.softmax(1)

    w_p0 = w_p0.clone()
    w_p0.requires_grad_()
    p0 = w_p0.softmax(0)

    g = avg_gain(rewards, mms, reset, eps, pi, p0)
    g.backward()

    return collections.namedtuple("Return", "gain, pi_grad, p0_grad")(g.item(), w_pi.grad.clone(), w_p0.grad.clone())
