# pylint: disable=no-member, invalid-name, not-callable, missing-docstring, line-too-long
import argparse
# import math
import os
import subprocess
from functools import partial
from time import perf_counter
import pickle

import torch

from bandit import grad_fn, init, master_matrix, steadystate, transfer_matrix
from gradientflow import flow


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
        if post.gain < pre.gain:
            return 2
        dpi = (pre.pi_grad - post.pi_grad).abs().max()
        dp0 = (pre.p0_grad - post.p0_grad).abs().max()
        if not torch.isfinite(dpi + dp0):
            return 0.0
        return (dpi + dp0).item() / max_dgrad

    for state, internals in flow(x, prepare, make_step, compare):
        yield state, internals


def optimize(args, w_pi, w_p0, mms, rewards, stop_steps, prefix=""):
    wall = perf_counter()
    wall_print = perf_counter()
    wall_save = perf_counter()

    dynamics = []
    t = 0

    for s, internals in flow_ode((w_pi, w_p0), partial(grad_fn, rewards, mms, args['reset'], args['eps']), max_dgrad=args['max_dgrad']):

        s['wall'] = perf_counter() - wall
        s['ngrad'] = internals['data'].pi_grad.abs().max().item()
        s['gain'] = internals['data'].gain
        s['loss'] = 1 - internals['data'].gain / args['gamma']

        save = False
        stop = False

        if perf_counter() - wall_save > 10:
            wall_save = perf_counter()
            save = True

        if s['step'] == stop_steps:
            save = True
            stop = True

        if s['t'] >= t or stop:
            t = 1.05 * s['t']
            dynamics.append(s)

        if perf_counter() - wall_print > 2 or save:
            wall_print = perf_counter()
            print(f"{prefix}wall={s['wall']:.0f} step={s['step']} t=({s['t']:.1e})+({s['dt']:.0e}) |dw|={s['ngrad']:.1e} loss={s['loss']:.3f}", flush=True)

        r = {
            'dynamics': dynamics,
            'w_pi': internals['x'][0],
            'w_p0': internals['x'][1],
            'pi': internals['x'][0].softmax(1),
            'p0': internals['x'][1].softmax(0),
            'stop': stop,
        }

        if save:
            yield r

        if stop:
            return


def last(i):
    x = None
    for x in i:
        pass
    return x


def optimal_u(r, mu, m):
    Sqrt = lambda x: x**0.5
    w1 = (-1 + Sqrt(1 + (-1 + r)**2*(-1 + mu**2)))/((-1 + r)*(1 + mu))
    w2 = -((1 + Sqrt(1 + (-1 + r)**2*(-1 + mu**2)))/((-1 + r)*(1 + mu)))
    eps = ((-1 + w1)*w1**m*(-1 + w2)*w2**m*(-1 + w1*w2)*(w1**m*w2 - w1*w2**m)**2 - Sqrt(-((-1 + w1)*w1**(1 + 2*m)*(-1 + w2)*w2**(1 + 2*m)*(-1 + w1*w2)**2*(w1**m - w2**m)*(w1**(3*m)*(1 + w1)*w2**2*(1 + w2) - w1**2*(1 + w1)*w2**(3*m)*(1 + w2) - w1**(2*m)*w2**(1 + m)*(2*w1 + w2 + w1*(7 + 2*w1)*w2 + (-1 + w1)*w2**2) + w1**(1 + m)*w2**(2*m)*(w1 + w1**2*(-1 + w2) + 2*w2 + w1*w2*(7 + 2*w2))))))/(w1**m*w2**m*(-1 + w1*w2)*(w1**2*w2**(2*m)*(1 + w1*(-1 + w2) + w2) + w1**(2*m)*w2**2*(1 + w1 + (-1 + w1)*w2) - 2*w1**(1 + m)*w2**(1 + m)*(1 + w1*w2)))

    q = -((-1 + w1)*w1**(2*m)*w2*(1 + w2)*(1 + w2*(-1 + eps))*(-1 + w1 + eps) + w1*(1 + w1)*(-1 + w2)*w2**(2*m)*(1 + w1*(-1 + eps))*(-1 + w2 + eps) - w1**m*w2**m*(-1 + w1*w2)*(-((-1 + w1)*(-1 + w2)*(w1 + w2)) + (-1 + w1)*(-1 + w2)*(w1 + w2)*eps + 2*w1*w2*eps**2))/(2.*(w1 - w2)*(-(w1**(2*m)*w2*(1 + w2*(-1 + eps))*(-1 + w1 + eps)) + w1*w2**(2*m)*(1 + w1*(-1 + eps))*(-1 + w2 + eps)))
    return eps, q


def w_pi_p0(args, states, actions, n_init_states):
    if args['init'] == 'randn':
        pi = torch.randn(len(states), len(actions), device=args['device']).mul(args['std0'])
        p0 = torch.randn(n_init_states, device=args['device']).mul(args['std0'])
        return pi, p0

    if args['init'] == 'optimal_u':
        assert args['memory_type'] == 'ram'
        e, _ = optimal_u(args['reset'], args['gamma'], args['memory'])
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
        assert args['memory_type'] == 'actions'
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
        [0.5 - args['gamma']/2, 0.5 + args['gamma']/2],
        [0.5 + args['gamma']/2, 0.5 - args['gamma']/2],
    ])

    mms = [master_matrix(states, actions, partial(prob, f)).to(device=args['device']) for f in fs]

    trials_steps = args['trials_steps']
    rs = [last(optimize(args, *w_pi_p0(args, states, actions, n_init_states), mms, rewards, trials_steps, prefix="TRIAL{}/{} ".format(i, args['trials']))) for i in range(args['trials'])]

    while len(rs) > 1:
        rs = sorted(rs, key=lambda r: r['dynamics'][-1]['gain'])
        print('best gain = {:.3f}'.format(rs[-1]['dynamics'][-1]['gain']))
        rs = rs[len(rs) // 2:]
        if len(rs) == 1:
            break
        trials_steps = round(trials_steps * 1.5)
        rs = [last(optimize(args, r['w_pi'], r['w_p0'], mms, rewards, trials_steps, prefix="TRIAL{}/{} ".format(i, len(rs)))) for i, r in enumerate(rs)]

    for r in optimize(args, rs[0]['w_pi'], rs[0]['w_p0'], mms, rewards, args['stop_steps']):
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
            'steadystates': [steadystate(transfer_matrix(r['pi'], mm, args['reset'], r['p0']), args['eps']) for mm in mms],
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
    parser.add_argument("--gamma", type=float, default=0.1)

    parser.add_argument("--init", type=str, required=True)
    parser.add_argument("--memory_type", type=str, required=True)
    parser.add_argument("--memory", type=int, required=True)
    parser.add_argument("--reset", type=float, required=True)

    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--max_dgrad", type=float, default=1e-4)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--eps_init", type=float, default=1e-4)
    parser.add_argument("--std0", type=float, default=1)

    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--trials_steps", type=int, default=0)

    parser.add_argument("--stop_steps", type=int, default=5000)
    # parser.add_argument("--stop_ngrad", type=float, default=1e-8)

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
