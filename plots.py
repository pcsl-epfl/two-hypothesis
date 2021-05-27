import numpy as np
import argparse
import math
from itertools import count, islice
from math import sqrt

import matplotlib.pyplot as plt
import torch

from grid import exec_grid, load_grouped
from main import optimal_u

torch.set_default_dtype(torch.float64)


def texnum(x, mfmt='{}', show_one=True):
    m, e = "{:e}".format(x).split('e')
    m, e = float(m), int(e)
    mx = mfmt.format(m)
    if float(mx) >= 10.0:
        m /= 10
        e += 1
        mx = mfmt.format(m)
    if e == 0:
        if m == 1:
            return "1" if show_one else ""
        return mx
    ex = r"10^{{{}}}".format(e)
    if m == 1:
        return ex
    return r"{}\;{}".format(mx, ex)


def pishow(pi, states, actions, p0=None, ss=None, eps=1e-5):

    if p0 is not None:
        x = torch.zeros(len(pi))
        x[:len(p0)] = p0
        pi = torch.cat([pi, x[:, None]], dim=1)
        actions = actions + ('p0',)

    if ss is not None:
        pi = torch.cat([pi, ss], dim=1)
        actions = actions + ('A', 'B')

    plt.figure(figsize=(0.11 * len(actions) + 1, 0.11 * len(states) + 0.5), dpi=100)
    pi = pi.clone()
    pi[pi < eps] = math.nan
    plt.imshow(pi, cmap="Oranges", vmin=0, vmax=1)
    plt.xticks(list(range(len(actions))), actions)
    plt.yticks(list(range(len(states))), states)
    plt.ylim(len(states) - 0.5, -0.5)
    plt.xlim(-0.5, len(actions) - 0.5)
    plt.grid()

    for tick in plt.gca().get_xticklabels():
        tick.set_fontname("monospace")
        tick.set_rotation(-90)
    for tick in plt.gca().get_yticklabels():
        tick.set_fontname("monospace")
        tick.set_fontsize(9)

    plt.tight_layout()


def sample(d):
    out = []
    t = 0
    for x in d:
        if x['t'] >= t:
            out.append(x)
            t = 1.1 * x['t']
    return out


def interp_median(xs, ys):
    x1 = max(x[0] for x in xs)
    x2 = min(x[-1] for x in xs)
    x = np.linspace(x1, x2, 1000)
    ys = [np.interp(x, xp, yp) for xp, yp in zip(xs, ys)]
    y = np.array(ys)
    return x, np.median(y, 0)


def is_prime(n):
    return n > 1 and all(n % i for i in islice(count(2), int(sqrt(n)-1)))


def plot_fig4(wall, python, threads):
    fig, [[ax11, ax12, ax13], [ax21, ax22, ax23]] = plt.subplots(2, 3, figsize=(5.5, 5), dpi=120, sharex=True, sharey=True)

    def plot1(data, memory_type, memory, init, reset, color=None, **kwargs):
        exec_grid(
            data,
            f"{python} main.py --stop_wall {wall} --stop_t 1e9 --mu 0.1",
            [
                ("init", [init]),
                ("memory_type", [memory_type]),
                ("memory", [memory]),
                ("reset", [reset]),
                ("seed", [i for i in range(20)]),
            ],
            n=threads,
        )

        def pred_args(a):
            return a['memory_type'] == memory_type and a['memory'] == memory and a['init'] == init and a['reset'] == reset

        args, groups = load_grouped(
            data,
            group_by=['seed', 'stop_t', 'stop_wall', 'stop_steps'],
            pred_args=pred_args
        )
        assert len(groups) == 1, groups[0][0].keys()

        for a, rs in groups:
            for r in rs:
                d = sample(r['dynamics'])
                [line] = plt.plot(
                    [x['t'] for x in d],
                    [x['q'] for x in d],
                    color=color,
                    alpha=0.1
                )
                color = line.get_color()

            ds = [sample(r['dynamics']) for r in rs]
            t, q = interp_median(
                [[np.log(x['t']) for x in d if x['t'] > 0.0] for d in ds],
                [[np.log(x['q']) for x in d if x['t'] > 0.0] for d in ds]
            )
            plt.plot(np.exp(t), np.exp(q), color=color, **kwargs)

        return args

    plt.sca(ax11)
    for init, label in [
        ('randn', 'RAM random'),
        ('randn_lin', 'RAM linear'),
        ('randn_u', 'RAM c.c.'),
        ('optimal_u', 'RAM optimal'),
    ]:
        plot1('glassy', memory_type='ram', memory=8, init=init, label=label, reset=1e-3)

    plt.sca(ax21)
    for init, label in [
        ('randn', 'RAM random'),
        ('randn_lin', 'RAM linear'),
        ('randn_u', 'RAM c.c.'),
        ('optimal_u', 'RAM optimal'),
    ]:
        plot1('glassy', memory_type='ram', memory=20, init=init, label=label, reset=1e-3)

    plt.sca(ax12)
    for init, label in [
        ('randn', 'Memento random'),
        ('randn_cycles', 'Memento cycles'),
    ]:
        plot1('glassy', memory_type='memento', memory=3, init=init, label=label, reset=1e-3)

    plt.sca(ax22)
    for init, label in [
        ('randn', 'Memento random'),
        ('randn_cycles', 'Memento cycles'),
    ]:
        plot1('glassy', memory_type='memento', memory=4, init=init, label=label, reset=1e-3)

    plt.sca(ax13)
    plot1('glassy', memory_type='memento', memory=3, init='randn', label='Memento random', reset=1e-5)
    args = plot1('glassy', memory_type='ram', memory=16, init='randn', label='RAM random', reset=1e-5)
    q, e = optimal_u(args['reset'], args['mu'], args['memory'])
    plt.plot([0, 1e13], [q, q], '--k', label='optimal')

    plt.sca(ax23)
    plot1('glassy', memory_type='memento', memory=4, init='randn', label='Memento random', reset=1e-5)
    args = plot1('glassy', memory_type='ram', memory=64, init='randn', label='RAM random', reset=1e-5)
    q, e = optimal_u(args['reset'], args['mu'], args['memory'])
    plt.plot([1e0, 1e19], [q, q], '--k', label='optimal')

    for ax in [ax11, ax21, ax12, ax22, ax13, ax23]:
        plt.sca(ax)
        plt.legend(
            loc=3,
            handlelength=1,
            labelspacing=0.2,
            handletextpad=0.4,
            frameon=False,
        )

        plt.plot([1e0, 1e9], [0, 0], 'k', linewidth=0.3)

    plt.ylim(-0.3, 0.55)
    plt.xscale('log')
    plt.xlim(1e0, 1e9)
    plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    plt.xticks([1e0, 1e3, 1e6])

    for ax in [ax11, ax21]:
        plt.sca(ax)
        plt.ylabel('$q$')
    for ax in [ax21, ax22, ax23]:
        plt.sca(ax)
        plt.xlabel('$t$')

    plt.sca(ax11)
    plt.title('RAM $M=8, r=10^{-3}$', fontsize=8)
    plt.annotate("A", (1e1, 0.4))

    plt.sca(ax21)
    plt.title('RAM $M=20, r=10^{-3}$', fontsize=8)
    plt.annotate("B", (1e1, 0.4))

    plt.sca(ax12)
    plt.title('Memento $m=3, r=10^{-3}$', fontsize=8)
    plt.annotate("C", (1e1, 0.4))

    plt.sca(ax22)
    plt.title('Memento $m=4, r=10^{-3}$', fontsize=8)
    plt.annotate("D", (1e1, 0.4))

    plt.sca(ax13)
    plt.title(r'$M_{\mathrm{eff}}=64, r=10^{-5}$', fontsize=8)
    plt.annotate("E", (1e1, 0.4))

    plt.sca(ax23)
    plt.title(r'$M_{\mathrm{eff}}=256, r=10^{-5}$', fontsize=8)
    plt.annotate("F", (1e1, 0.4))

    plt.tight_layout(h_pad=0.2, w_pad=-3)
    plt.savefig('fig4.png')


def plot_fig2(wall, python, threads):
    exec_grid(
        "ram_opt_reset",
        f"{python} main.py --memory_type ram --stop_wall {wall} --stop_t 1e9 --init optimal_u",
        [
            ("memory", [5, 10, 20]),
            ("mu", [0.1, 0.2]),
            ("reset", [2**i for i in range(-20, 0)]),
        ],
        n=threads,
    )
    exec_grid(
        "ram_opt_mem",
        f"{python} main.py --memory_type ram --stop_wall {wall} --stop_t 1e9 --init optimal_u",
        [
            ("mu", [0.1, 0.2]),
            ("reset", [1e-6, 1e-2, 1e-4]),
            ("memory", [2, 3, 4, 5, 7, 9, 11, 14, 17, 20, 23, 26, 29, 33, 37]),
        ],
        n=threads,
    )

    fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(5.5, 2.5), dpi=100, sharey=True)

    plt.sca(ax1)
    args, groups = load_grouped(
        'ram_opt_reset',
        group_by=['seed', 'reset', 'stop_wall']
    )

    for a, rs in sorted(groups, key=lambda x: x[0]['mu']):
        resets = sorted({r['args']['reset'] for r in rs})

        color = {
            (0.1, 10): '#7e4bdd',
            (0.1, 20): '#A91BD1',
            (0.1, 5): '#E817B7',
            (0.2, 10): '#68DD74',
            (0.2, 20): '#1E6625',
            (0.2, 5): '#B6D657',
        }[(a['mu'], a['memory'])]

        [line] = plt.plot(
            resets,
            [
                min([r['dynamics'][-1]['q'] for r in rs if r['args']['reset'] == re])
                for re in resets
            ],
            '.',
            label=fr"$\mu={a['mu']} \quad M={a['memory']}$",
            color=color,
        )
        r = torch.logspace(-7, 0, 100)
        mu = a['mu']
        m = a['memory']
        plt.plot(r, optimal_u(r, mu, m)[1], color=line.get_color())

    plt.xlim(min(r['args']['reset'] for r in rs), 1)
    plt.legend(handlelength=0.5, labelspacing=0)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('$r$')
    plt.ylabel('$q$')

    plt.sca(ax2)
    args, groups = load_grouped('ram_opt_mem', group_by=['seed', 'memory', 'stop_wall'])

    for a, rs in sorted(groups, key=lambda x: x[0]['mu']):
        mems = sorted({r['args']['memory'] for r in rs})

        color = {
            (0.1, 1e-6): '#7e4bdd',
            (0.1, 1e-4): '#A91BD1',
            (0.1, 1e-2): '#E817B7',
            (0.2, 1e-6): '#68DD74',
            (0.2, 1e-4): '#1E6625',
            (0.2, 1e-2): '#B6D657',
        }[(a['mu'], a['reset'])]

        [line] = plt.plot(
            mems,
            [
                min([r['dynamics'][-1]['q'] for r in rs if r['args']['memory'] == m])
                for m in mems
            ],
            '.',
            label=fr"$\mu={a['mu']} \quad r={texnum(a['reset'])}$",
            color=color,
        )
        m = torch.logspace(0, 2, 100)
        mu = a['mu']
        r = a['reset']
        plt.plot(m, optimal_u(r, mu, m)[1], color=line.get_color())

    plt.xlim(1, max(r['args']['memory'] for r in rs))
    plt.legend(handlelength=0.5, labelspacing=-0.2)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('$M$')

    plt.tight_layout(pad=1)
    plt.savefig('fig2.png')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", type=str, default='python')
    parser.add_argument("--wall", type=float, default=120)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args().__dict__

    plot_fig4(args['wall'], args['python'], args['threads'])
    plot_fig2(args['wall'], args['python'], args['threads'])
