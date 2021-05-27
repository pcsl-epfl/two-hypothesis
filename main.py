import numpy  # noqa
import argparse
import collections
import itertools
import os
import pickle
import subprocess
from functools import partial
from time import perf_counter

import torch
from gradientflow import flow


def str_prod(*iterators, n=1):
    return tuple([''.join(x) for x in itertools.product(*iterators, repeat=n)])


def init(n_arms, mem, mem_type):
    arms = tuple('ABCDEFGH'[:n_arms])

    n_init_states = 0

    if mem_type == 'ram':
        mem_states = [f"{i:02d}" for i in range(mem)]
        actions = str_prod(mem_states, arms)
        states = str_prod(mem_states, ['  ']) + str_prod(mem_states, arms, '+-')
        n_init_states = len(mem_states)
        rewards = torch.tensor([{'+': +1.0, '-': -1.0, ' ': 0.0}[s[-1]] for s in states])
        def prob(f, ss, _s, a):
            # s = 44A+
            # a = 32B
            #ss = 32B-
            if ss[:-1] == a:
                fa = f[arms.index(a[-1])]
                return fa if ss[-1] == '+' else 1 - fa
            return 0

    elif mem_type == 'memento':
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
    for _ in itertools.count():
        m = m @ m

        if (m - m[:, [0]]).abs().max().item() < eps:
            return m[:, 0]

        if not torch.isfinite(m).all():
            return m[:, 0]


def avg_exp_reward(rewards, mms, reset, eps, pi, p0):
    g = 0
    for mm in mms:
        m = transfer_matrix(pi, mm, reset, p0)
        p = steadystate(m, eps)
        g += torch.dot(rewards, p)
    g = g / len(mms)
    return g


def grad_fn(rewards, mms, reset, eps, w_pi, w_p0):
    w_pi = w_pi.clone()
    w_pi.requires_grad_()
    pi = w_pi.softmax(1)

    w_p0 = w_p0.clone()
    w_p0.requires_grad_()
    p0 = w_p0.softmax(0)

    g = avg_exp_reward(rewards, mms, reset, eps, pi, p0)
    g.backward()

    return collections.namedtuple("Return", "exp_reward, pi_grad, p0_grad")(g.item(), w_pi.grad.clone(), w_p0.grad.clone())


def flow_ode(x, grad_fun, max_dgrad):
    """
    flow for an ODE
    """

    def prepare(xx, t, post, post_t):
        if post is not None and post_t == t:
            return post
        w_pi, w_p0 = xx
        return grad_fun(w_pi, w_p0)

    def make_step(xx, pre, _t, dt):
        w_pi, w_p0 = xx
        return (w_pi + dt * pre.pi_grad, w_p0 + dt * pre.p0_grad)

    def compare(pre, post):
        if post.exp_reward < pre.exp_reward:
            return 2
        dpi = (pre.pi_grad - post.pi_grad).abs().max()
        dp0 = (pre.p0_grad - post.p0_grad).abs().max()
        if not torch.isfinite(dpi + dp0):
            return 0.0
        return (dpi + dp0).item() / max_dgrad

    for state, internals in flow(x, prepare, make_step, compare):
        yield state, internals


def optimize(args, w_pi, w_p0, mms, rewards, prefix=""):
    wall = perf_counter()
    wall_print = perf_counter()
    wall_save = perf_counter()

    dynamics = []
    t = 0
    last_saved_q = 0.5

    for s, internals in flow_ode((w_pi, w_p0), partial(grad_fn, rewards, mms, args['reset'], args['eps_power']), max_dgrad=args['max_dgrad']):

        s['wall'] = perf_counter() - wall
        s['ngrad'] = internals['data'].pi_grad.abs().max().item()
        s['exp_reward'] = internals['data'].exp_reward
        s['gain'] = args['reset'] * s['exp_reward']
        s['q'] = (1 - internals['data'].exp_reward / args['mu']) / 2

        save = False
        stop = False

        if s['step'] == 0:
            save = True

        if perf_counter() - wall_save > 10:
            save = True

        if s['q'] < 0.98 * last_saved_q:
            save = True

        if s['step'] == args['stop_steps']:
            save = True
            stop = True

        if args['stop_t'] is not None and s['t'] >= args['stop_t']:
            save = True
            stop = True

        if args['stop_wall'] is not None and s['wall'] >= args['stop_wall']:
            save = True
            stop = True

        if s['t'] >= t or stop or save:
            t = 1.05 * s['t']
            dynamics.append(s)

        if perf_counter() - wall_print > 2 or save:
            wall_print = perf_counter()
            print(f"{prefix}wall={s['wall']:.0f} step={s['step']} t=({s['t']:.1e})+({s['dt']:.0e}) |dw|={s['ngrad']:.1e} q={s['q']:.5f}", flush=True)

        if save:
            wall_save = perf_counter()
            last_saved_q = s['q']

            yield {
                'dynamics': dynamics,
                'w_pi': internals['x'][0],
                'w_p0': internals['x'][1],
                'pi': internals['x'][0].softmax(1),
                'p0': internals['x'][1].softmax(0),
                'stop': stop,
            }

        if stop:
            return


def last(i):
    x = None
    for x in i:
        pass
    return x


def optimal_u(r, mu, m):
    try:
        Sqrt = lambda x: x**0.5
        w1 = (-1 + Sqrt(1 + (-1 + r)**2*(-1 + mu**2)))/((-1 + r)*(1 + mu))
        w2 = -((1 + Sqrt(1 + (-1 + r)**2*(-1 + mu**2)))/((-1 + r)*(1 + mu)))
        eps = ((-1 + w1)*w1**m*(-1 + w2)*w2**m*(-1 + w1*w2)*(w1**m*w2 - w1*w2**m)**2 - Sqrt(-((-1 + w1)*w1**(1 + 2*m)*(-1 + w2)*w2**(1 + 2*m)*(-1 + w1*w2)**2*(w1**m - w2**m)*(w1**(3*m)*(1 + w1)*w2**2*(1 + w2) - w1**2*(1 + w1)*w2**(3*m)*(1 + w2) - w1**(2*m)*w2**(1 + m)*(2*w1 + w2 + w1*(7 + 2*w1)*w2 + (-1 + w1)*w2**2) + w1**(1 + m)*w2**(2*m)*(w1 + w1**2*(-1 + w2) + 2*w2 + w1*w2*(7 + 2*w2))))))/(w1**m*w2**m*(-1 + w1*w2)*(w1**2*w2**(2*m)*(1 + w1*(-1 + w2) + w2) + w1**(2*m)*w2**2*(1 + w1 + (-1 + w1)*w2) - 2*w1**(1 + m)*w2**(1 + m)*(1 + w1*w2)))
        q = -((-1 + w1)*w1**(2*m)*w2*(1 + w2)*(1 + w2*(-1 + eps))*(-1 + w1 + eps) + w1*(1 + w1)*(-1 + w2)*w2**(2*m)*(1 + w1*(-1 + eps))*(-1 + w2 + eps) - w1**m*w2**m*(-1 + w1*w2)*(-((-1 + w1)*(-1 + w2)*(w1 + w2)) + (-1 + w1)*(-1 + w2)*(w1 + w2)*eps + 2*w1*w2*eps**2))/(2.*(w1 - w2)*(-(w1**(2*m)*w2*(1 + w2*(-1 + eps))*(-1 + w1 + eps)) + w1*w2**(2*m)*(1 + w1*(-1 + eps))*(-1 + w2 + eps)))
    except ZeroDivisionError:
        eps = 0.0
        q = 1.0
    return eps, q


def w_pi_p0(args, states, actions, n_init_states):
    if args['init'] == 'randn':
        pi = torch.randn(len(states), len(actions), device=args['device']).mul(args['std0'])
        p0 = torch.randn(n_init_states, device=args['device']).mul(args['std0'])
        return pi, p0

    if args['init'] == 'optimal_u':
        assert args['memory_type'] == 'ram'
        e, _ = optimal_u(args['reset'], args['mu'], args['memory'])
        e = min(e, 1)
        pi = torch.zeros(len(states), len(actions), device=args['device'])
        for i, s in enumerate(states):
            # s = 44A+
            # a = 32B
            x = int(s[:2])

            if s[2:] == "  ":
                continue

            if x == args['memory'] - 1:
                if s[-1] == "+":
                    pi[i, actions.index(s[:-1])] = 1.0
                else:
                    pi[i, actions.index(s[:-1])] = 1.0 - e
                    pi[i, actions.index(f"{x-1:02d}{s[-2]}")] = e

            elif x == 0:
                if s[-1] == "+":
                    pi[i, actions.index(f"01{s[-2]}")] = 1.0
                elif s[-2] == "B":
                    pi[i, actions.index("00A")] = 1.0
                elif s[-2] == "A":
                    pi[i, actions.index("00B")] = 1.0

            else:
                if s[-1] == "+":
                    pi[i, actions.index(f"{x+1:02d}{s[-2]}")] = 1.0
                if s[-1] == "-":
                    pi[i, actions.index(f"{x-1:02d}{s[-2]}")] = 1.0

        pi[states.index("00  "), actions.index("00A")] = 0.5
        pi[states.index("00  "), actions.index("00B")] = 0.5

        p0 = torch.zeros(n_init_states, device=args['device'])
        p0[states.index("00  ")] = 1
        return (pi + args['eps_init']).log(), (p0 + args['eps_init']).log()

    if args['init'] == 'randn_lin':
        assert args['memory_type'] == 'ram'
        pi = torch.randn(len(states), len(actions), device=args['device']).mul(args['std0'])
        pi = pi.softmax(1)
        for i, s in enumerate(states):
            for j, a in enumerate(actions):
                # s = '44  '
                # s = '44A+'
                # a = '32B'
                if abs(int(s[:2]) - int(a[:2])) > 1:
                    pi[i, j] = 0.0

        p0 = torch.randn(n_init_states, device=args['device']).mul(args['std0'])
        pi = (pi / pi.sum(1, True) + args['eps_init']).log()
        assert torch.isfinite(pi).all()
        return pi, p0

    if args['init'] == 'randn_u':
        assert args['memory_type'] == 'ram'
        pi = torch.randn(len(states), len(actions), device=args['device']).mul(args['std0'])
        pi = pi.softmax(1)
        for i, s in enumerate(states):
            for j, a in enumerate(actions):
                # s = '44  '
                # s = '44A+'
                # a = '32B'
                if abs(int(s[:2]) - int(a[:2])) > 1:
                    pi[i, j] = 0.0

                if s[2] != ' ' and s[2] != a[2] and int(s[:2]) > 0:
                    pi[i, j] = 0.0

        p0 = torch.randn(n_init_states, device=args['device']).mul(args['std0'])
        pi = (pi / pi.sum(1, True) + args['eps_init']).log()
        assert torch.isfinite(pi).all()
        return pi, p0

    if args['init'] == 'randn_cycles':
        assert args['memory_type'] == 'memento'
        pi = torch.randn(len(states), len(actions), device=args['device']).mul(args['std0'])
        pi = pi.softmax(1)
        m = args['memory']

        for i, s in enumerate(states):
            # s = '      '
            # s = '  A  -'
            # s = 'AAB++-'
            # a = B
            if s[0] == ' ':
                continue

            x = torch.tensor([{'A': 1.0, 'B': -1.0, '+': 1.0, '-': -1.0, ' ': 0.0}[i] for i in s])
            # if not full red flag
            if (x[:m] * x[m:]).var() > 0.0:
                pi[i] = 0.0
                pi[i, actions.index(s[0])] = 1.0

        p0 = torch.randn(n_init_states, device=args['device']).mul(args['std0'])
        return (pi + args['eps_init']).log(), p0


def execute(args):
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(args['seed'])

    states, actions, arms, rewards, prob, n_init_states = init(n_arms=args['arms'], mem=args['memory'], mem_type=args['memory_type'])
    rewards = rewards.to(device=args['device'])

    assert args['arms'] == 2
    fs = torch.tensor([
        [0.5 - args['mu']/2, 0.5 + args['mu']/2],
        [0.5 + args['mu']/2, 0.5 - args['mu']/2],
    ])

    mms = [master_matrix(states, actions, partial(prob, f)).to(device=args['device']) for f in fs]

    for r in optimize(args, *w_pi_p0(args, states, actions, n_init_states), mms, rewards):
        yield {
            'args': args,
            'states': states,
            'actions': actions,
            'arms': arms,
            'dynamics': r['dynamics'],
            'w_pi': r['w_pi'],
            'w_p0': r['w_p0'],
            'pi': r['pi'],
            'p0': r['p0'],
            'steadystates': [steadystate(transfer_matrix(r['pi'], mm, args['reset'], r['p0']), args['eps_power']) for mm in mms],
            'finished': r['stop']
        }


def main():
    git = {
        'log': subprocess.getoutput('git log --format="%H" -n 1 -z'),
        'status': subprocess.getoutput('git status -z'),
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default='cpu')

    parser.add_argument("--arms", type=int, default=2)
    parser.add_argument("--mu", type=float, default=0.1)

    parser.add_argument("--init", type=str, required=True)
    parser.add_argument("--memory_type", type=str, required=True)
    parser.add_argument("--memory", type=int, required=True)
    parser.add_argument("--reset", type=float, required=True)

    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--max_dgrad", type=float, default=1e-4)
    parser.add_argument("--eps_power", type=float, default=1e-8)
    parser.add_argument("--eps_init", type=float, default=1e-4)
    parser.add_argument("--std0", type=float, default=1)

    parser.add_argument("--stop_steps", type=int)
    parser.add_argument("--stop_wall", type=float)
    parser.add_argument("--stop_t", type=float)

    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args().__dict__

    with open(args['output'], 'wb') as handle:
        pickle.dump(args, handle)
    saved = False
    try:
        for data in execute(args):
            data['git'] = git
            with open(args['output'], 'wb') as handle:
                pickle.dump(args, handle)
                pickle.dump(data, handle)
            saved = True
    except:
        if not saved:
            os.remove(args['output'])
        raise


if __name__ == "__main__":
    main()
