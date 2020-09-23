# pylint: disable=no-member, invalid-name, not-callable, missing-docstring, line-too-long
import argparse
import os
import subprocess
from functools import partial
from time import perf_counter

import torch

from bandit import grad_fn, init, master_matrix
from gradientflow import gradientflow_ode


def prob(f, arms, ss, s, a):
    # s = +1101
    # a = A0
    #ss = -1010
    if ss[1:] == s[2:] + a[1]:
        fa = f[arms.index(a[0])]
        return fa if ss[0] == '+' else 1 - fa
    return 0


def execute(args):
    wall = perf_counter()
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(args.seed)

    states, actions, arms, rewards = init(n_arms=args.arms, mem=args.memory)
    rewards = rewards.to(device=args.device)

    fs = torch.tensor([
        [0.5, 0.5 + args.gamma],
        [0.5, 0.5 - args.gamma],
        [0.5 + args.gamma, 0.5],
        [0.5 - args.gamma, 0.5],
    ])

    mms = [master_matrix(states, actions, partial(prob, f, arms)).to(device=args.device) for f in fs]

    w = torch.randn(len(states), len(actions)).mul(1).to(device=args.device)
    dynamics = []

    wall_print = perf_counter()
    wall_save = perf_counter()

    for state, internals in gradientflow_ode(w, partial(grad_fn, rewards, mms, args.reset, args.eps), max_dgrad=args.max_dgrad):

        state['wall'] = perf_counter() - wall
        state['ngrad'] = internals['gradient'].norm().item()
        state['gain'] = internals['custom']
        dynamics.append(state)

        if perf_counter() - wall_print > 2:
            wall_print = perf_counter()
            print("wall={0[wall]:.0f} step={0[step]} t=({0[t]:.1e})+({0[dt]:.0e}) |dw|={0[ngrad]:.1e} G={0[gain]:.3f}".format(state), flush=True)

        save = False
        stop = False

        if perf_counter() - wall_save > 10:
            wall_save = perf_counter()
            save = True

        if state['step'] == args.step_stop:
            save = True
            stop = True

        if save:
            yield {
                'args': args,
                'states': states,
                'actions': actions,
                'arms': arms,
                'dynamics': dynamics,
                'pi': internals['variables'].softmax(1),
            }

        if stop:
            return


def main():
    git = {
        'log': subprocess.getoutput('git log --format="%H" -n 1 -z'),
        'status': subprocess.getoutput('git status -z'),
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default='cpu')

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--memory", type=int, required=True)
    parser.add_argument("--arms", type=int, required=True)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--reset", type=float, default=0.0)

    parser.add_argument("--max_dgrad", type=float, default=1e-4)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--step_stop", type=int, default=1000)

    parser.add_argument("--pickle", type=str, required=True)
    args = parser.parse_args()

    torch.save(args, args.pickle, _use_new_zipfile_serialization=False)
    saved = False
    try:
        for res in execute(args):
            res['git'] = git
            with open(args.pickle, 'wb') as f:
                torch.save(args, f, _use_new_zipfile_serialization=False)
                torch.save(res, f, _use_new_zipfile_serialization=False)
                saved = True
    except:
        if not saved:
            os.remove(args.pickle)
        raise


if __name__ == "__main__":
    main()
