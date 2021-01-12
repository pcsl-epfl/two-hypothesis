# pylint: disable=no-member, invalid-name, not-callable, missing-docstring, line-too-long
import argparse
# import math
import os
import subprocess
from functools import partial
from time import perf_counter
import pickle

import torch

from bandit import grad_fn, init, master_matrix
from gradientflow import flow


def flow_ode(x, grad_fun, max_dgrad=1e-4):
    """
    flow for an ODE
    """

    def prepare(xx, t, old_data, old_t):
        if old_data is not None and old_t == t:
            return old_data
        return grad_fun(xx)

    def make_step(xx, data, _t, dt):
        g, _ = data
        return xx + dt * g

    def compare(data1, data2):
        g1, _ = data1
        g2, _ = data2
        dgrad = (g1 - g2).abs().max()
        return dgrad.item() / max_dgrad

    for state, internals in flow(x, prepare, make_step, compare):
        yield state, internals


def optimize(args, states, w, mms, rewards, stop_steps, prefix=""):
    wall = perf_counter()
    wall_print = perf_counter()
    wall_save = perf_counter()

    dynamics = []

    for state, internals in flow_ode(w, partial(grad_fn, states, rewards, mms, args.reset, args.eps, args.ab_sym), max_dgrad=args.max_dgrad):

        state['wall'] = perf_counter() - wall
        state['ngrad'] = internals['data'][0].abs().max().item()
        state['gain'] = internals['data'][1]
        dynamics.append(state)

        if perf_counter() - wall_print > 2:
            wall_print = perf_counter()
            print("{1}wall={0[wall]:.0f} step={0[step]} t=({0[t]:.1e})+({0[dt]:.0e}) |dw|={0[ngrad]:.1e} G={0[gain]:.3f}".format(state, prefix), flush=True)

        save = False
        stop = False

        if perf_counter() - wall_save > 10:
            wall_save = perf_counter()
            save = True

        if state['ngrad'] < args.stop_ngrad:
            save = True
            stop = True

        if state['step'] == stop_steps:
            save = True
            stop = True

        r = {
            'dynamics': dynamics,
            'weights': internals['x'],
            'pi': internals['x'].softmax(1),
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


def execute(args):
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(args.seed)

    if args.trials_memory_type == args.memory_type:
        trials_memory = args.memory
    if args.trials_memory_type == 'shift' and args.memory_type == 'ram':
        trials_memory = args.memory
    if args.trials_memory_type == 'actions' and args.memory_type == 'ram':
        trials_memory = (args.memory + 1) // args.arms

    states, actions, arms, rewards, prob = init(n_arms=args.arms, mem=trials_memory, mem_type=args.trials_memory_type)
    rewards = rewards.to(device=args.device)

    assert args.arms == 2
    fs = torch.tensor([
        [0.5 - args.gamma/2, 0.5 + args.gamma/2],
        [0.5 + args.gamma/2, 0.5 - args.gamma/2],
    ])

    mms = [master_matrix(states, actions, partial(prob, f)).to(device=args.device) for f in fs]

    def w():
        if args.trials_memory_type == 'restless':
            pi = torch.zeros(len(states), len(actions), device=args.device).fill_(-1)

            for i, s in enumerate(states):
                for j, a in enumerate(actions):
                    if a[1] == '>' and (s[:2] in ["+B", "-A"]):
                        pi[i, j] = 1
                    if a[1] == '<' and (s[:2] in ["+A", "-B"]):
                        pi[i, j] = 1

            noise = pi.clone()
            noise.normal_().mul_(args.std0)
            return pi + noise
        return torch.randn(len(states) // 2 if args.ab_sym else len(states), len(actions), device=args.device).mul(args.std0)

    trials_steps = args.trials_steps
    rs = [last(optimize(args, states, w(), mms, rewards, trials_steps, prefix="TRIAL{}/{} ".format(i, args.trials))) for i in range(args.trials)]

    while len(rs) > 1:
        rs = sorted(rs, key=lambda r: r['dynamics'][-1]['gain'])
        print('best gain = {:.3f}'.format(rs[-1]['dynamics'][-1]['gain']))
        rs = rs[len(rs) // 2:]
        if len(rs) == 1:
            break
        trials_steps = round(trials_steps * 1.5)
        rs = [last(optimize(args, states, r['weights'], mms, rewards, trials_steps, prefix="TRIAL{}/{} ".format(i, len(rs)))) for i, r in enumerate(rs)]

    r = rs[0]

    states2, actions2, arms2, rewards2, prob2 = init(n_arms=args.arms, mem=args.memory, mem_type=args.memory_type)
    mms2 = [master_matrix(states2, actions2, partial(prob2, f)).to(device=args.device) for f in fs]
    rewards2 = rewards2.to(device=args.device)

    if args.memory_type == args.trials_memory_type:
        w2 = r['weights']

    if args.memory_type == 'ram' and args.trials_memory_type == 'shift':
        assert states2 == states
        assert arms2 == arms
        w2 = torch.ones(len(states2), len(actions2)).mul(-5)
        for s, line in zip(states, r['weights']):
            for a, x in zip(actions, line):
                a = a[0] + s[2:] + a[1]
                w2[states2.index(s), actions2.index(a)] = x

    if args.memory_type == 'ram' and args.trials_memory_type == 'actions':
        assert arms2 == arms
        w2 = torch.ones(len(states2), len(actions2)).mul(-5)
        for s, line in zip(states, r['weights']):
            for a, x in zip(actions, line):
                def f(s):
                    return s.replace('A', '0').replace('B', '1').replace('-', '0').replace('+', '1')
                # s = ABA++-
                # a = B
                m = len(s) // 2
                s2 = s[-1] + f(s[:-1])
                a2 = a + f(s[1:m]) + f(a) + f(s[m+1:])
                while len(s2) < args.memory + 1:
                    s2 = s2 + '0'
                while len(a2) < args.memory + 1:
                    a2 = a2 + '0'
                w2[states2.index(s2), actions2.index(a2)] = x

    for r2 in optimize(args, states, w2, mms2, rewards2, args.stop_steps):
        yield {
            'args': args,
            'states': states2,
            'actions': actions2,
            'arms': arms2,
            'dynamics': r2['dynamics'],
            'weights': r2['weights'],
            'pi': r2['pi'],
            'finished': r2['stop']
        }


def main():
    git = {
        'log': subprocess.getoutput('git log --format="%H" -n 1 -z'),
        'status': subprocess.getoutput('git status -z'),
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default='cpu')

    parser.add_argument("--memory_type", type=str, required=True)
    parser.add_argument("--memory", type=int, required=True)
    parser.add_argument("--arms", type=int, default=2)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--reset", type=float, default=0.0)

    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--max_dgrad", type=float, default=1e-4)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--std0", type=float, default=1)

    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--trials_steps", type=int, default=0)
    parser.add_argument("--trials_memory_type", type=str)

    parser.add_argument("--stop_steps", type=int, default=5000)
    parser.add_argument("--stop_ngrad", type=float, default=1e-8)
    parser.add_argument("--ab_sym", type=int, default=0)

    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    if args.trials_memory_type is None:
        args.trials_memory_type = args.memory_type

    with open(args.output, 'wb') as handle:
        pickle.dump(args, handle)
    saved = False
    try:
        for data in execute(args):
            data['git'] = git
            with open(args.output, 'wb') as handle:
                pickle.dump(args, handle)
                pickle.dump(data, handle)
            saved = True
    except:
        if not saved:
            os.remove(args.output)
        raise


if __name__ == "__main__":
    main()
