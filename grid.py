import datetime
import glob
import itertools
import os
import pickle
import random
import re
import shlex
import subprocess
import threading
import time
from collections import defaultdict, namedtuple
from itertools import count, product

try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:
    def tqdm(x):
        return x


Run = namedtuple('Run', 'file, time, args, data')
GLOBALCACHE = defaultdict(dict)


def deepmap(fun, data):
    if isinstance(data, (list, tuple, set, frozenset)):
        return type(data)(deepmap(fun, x) for x in data)

    if isinstance(data, dict):
        return {key: deepmap(fun, x) for key, x in data.items()}

    return fun(data)


def torch_to_numpy(data):
    import torch
    def fun(x):
        if isinstance(x, torch.Tensor):
            return x.numpy()
        else:
            return x
    return deepmap(fun, data)


def load(directory, pred_args=None, pred_run=None, cache=True, extractor=None, convertion=None):
    return list(load_iter(directory, pred_args, pred_run, cache, extractor, convertion))


def load_iter(directory, pred_args=None, pred_run=None, cache=True, extractor=None, convertion=None):
    if extractor is not None:
        cache = False

    directory = os.path.normpath(directory)

    if not os.path.isdir(directory):
        raise NotADirectoryError('{} does not exists'.format(directory))

    cache_runs = GLOBALCACHE[(directory, convertion)] if cache else dict()

    for file in tqdm(sorted(glob.glob(os.path.join(directory, '*.pk')))):
        time = os.path.getctime(file)

        if file in cache_runs and time == cache_runs[file].time:
            x = cache_runs[file]

            if pred_args is not None and not pred_args(x.args):
                continue

            if pred_run is not None and not pred_run(x.data):
                continue

            yield x.data
            continue

        with open(file, 'rb') as f:
            try:
                args = pickle.load(f)

                if pred_args is not None and not pred_args(args):
                    continue

                data = pickle.load(f)
            except:
                continue

        if extractor is not None:
            data = extractor(data)

        if convertion == 'torch_to_numpy':
            data = torch_to_numpy(data)

        x = Run(file=file, time=time, args=args, data=data)
        cache_runs[file] = x

        if pred_run is not None and not pred_run(x.data):
            continue

        yield x.data


def hashable(x):
    if isinstance(x, list):
        x = tuple(hashable(i) for i in x)
    if isinstance(x, set):
        x = frozenset(x)
    try:
        hash(x)
    except TypeError:
        return '<not hashable>'
    return x


def keyall(x):
    if x is None:
        return (0, x)
    if isinstance(x, bool):
        return (1, x)
    if isinstance(x, str):
        return (2, x)
    if isinstance(x, (int, float)):
        return (3, x)
    if isinstance(x, tuple):
        return (4, tuple(keyall(i) for i in x))
    if isinstance(x, list):
        return (5, [keyall(i) for i in x])
    return (6, x)


def args_intersection(argss):
    return {k: list(v)[0] for k, v in args_union(argss).items() if len(v) == 1}


def args_todict(r):
    if not isinstance(r, dict):
        r = r.__dict__
    return {
        key: hashable(value)
        for key, value in r.items()
        if key not in ['pickle', 'output']
    }


def args_union(argss):
    argss = [args_todict(r) for r in argss]
    keys = {key for r in argss for key in r.keys()}

    return {
        key: {r[key] if key in r else None for r in argss}
        for key in keys
    }


def args_diff(argss):
    args = args_intersection(argss)
    argss = [args_todict(r) for r in argss]
    return [
        {
            key: a[key]
            for key in a.keys()
            if key not in args.keys()
        }
        for a in argss
    ]


def get_args_item(args, key):
    if hasattr(args, key):
        return getattr(args, key)
    if isinstance(args, dict) and key in args:
        return args[key]
    return None


def load_grouped(directory, group_by, pred_args=None, pred_run=None, convertion=None):
    """

    example:

    args, groups = load_grouped('results', ['alpha', 'seed_init'])

    for param, rs in groups:
        # in `rs` only 'alpha' and 'seed_init' can vary
        plot(rs, label=param)

    """
    runs = load(directory, pred_args=pred_args, pred_run=pred_run, convertion=convertion)

    return group_runs(runs, group_by)


def group_runs(runs, group_by):
    args = args_intersection([r['args'] for r in runs])
    variants = {
        key: sorted(values, key=keyall)
        for key, values in args_union([r['args'] for r in runs]).items()
        if len(values) > 1 and key not in group_by
    }

    groups = []
    for vals in itertools.product(*variants.values()):
        var = {k: v for k, v in zip(variants, vals)}

        rs = [
            r
            for r in runs
            if all(
                hashable(get_args_item(r['args'], k)) == v
                for k, v in var.items()
            )
        ]
        if rs:
            groups.append((var, rs))
    assert len(runs) == sum(len(rs) for _a, rs in groups)

    return args, groups


def load_args(f):
    for _ in range(5):
        try:
            with open(f, 'rb') as rb:
                return pickle.load(rb)
        except:
            time.sleep(0.1)
    with open(f, 'rb') as rb:
        return pickle.load(rb)


def to_dict(x):
    if isinstance(x, dict):
        return x

    return x.__dict__


def load_data(f):
    for _ in range(5):
        try:
            with open(f, 'rb') as rb:
                pickle.load(rb)
                return pickle.load(rb)
        except:
            time.sleep(0.1)
    with open(f, 'rb') as rb:
        pickle.load(rb)
        return pickle.load(rb)


def print_output(out, text, path):
    if path is not None:
        open(path, 'ta').close()

    for line in iter(out.readline, b''):
        output = line.decode("utf-8")
        m = re.findall(r"job (\d+)", output)  # srun: job (\d+) has been allocated resources
        if m and len(text) < 2:
            text.insert(0, m[0])

        print("[{}] {}".format(" ".join(text), output), end="")

        if path is not None:
            with open(path, 'ta') as f:
                f.write("{} [{}] {}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), " ".join(text), line.decode("utf-8")))

    if path is None:
        print("[{}] terminated".format(" ".join(text)))


def exec_grid(log_dir, cmd, params, sleep=0, n=None):
    command = "{} --output {{output}}".format(cmd)

    for name, _vals in params:
        command += " --{0} {{{0}}}".format(name)

    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)

    with open(os.path.join(log_dir, "info"), 'wb') as f:
        pickle.dump({
            'cmd': cmd,
            'params': params,
            'git': {
                'log': subprocess.getoutput('git log --format="%H" -n 1 -z'),
                'status': subprocess.getoutput('git status -z'),
            }
        }, f)

    done_files = set()
    done_param = dict()

    for f in tqdm(glob.glob(os.path.join(log_dir, "*.pk"))):
        if f not in done_files:
            done_files.add(f)

            a = to_dict(load_args(f))
            a = tuple((name, a[name] if name in a else None) for name, _vals in params)
            done_param[a] = f

    running = []
    threads = []

    for param in product(*[vals for name, vals in params]):
        param = tuple((name, val) for val, (name, vals) in zip(param, params))

        if len(running) > 0:
            time.sleep(sleep)

        if n is not None:
            while len(running) >= n:
                running = [x for x in running if x.poll() is None]
                time.sleep(0.2)

        if os.path.isfile('stop'):
            print()
            print('  >> stop file detected!  <<')
            print()
            break

        for f in glob.glob(os.path.join(log_dir, "*.pk")):
            if f not in done_files:
                done_files.add(f)

                a = to_dict(load_args(f))
                a = tuple((name, a[name] if name in a else None) for name, _vals in params)
                done_param[a] = f

        text = " ".join("{}={}".format(name, val) for name, val in param)

        if param in done_param:
            print('[{}] {}'.format(text, done_param[param]))
            continue

        for i in count(random.randint(0, 999_999)):
            i = i % 1_000_000
            fn = "{:06d}.pk".format(i)
            fp = os.path.join(log_dir, fn)
            if not os.path.isfile(fp):
                break

        text = "{} {}".format(fp, text)
        text = [text]

        cmd = command.format(output=fp, **dict(param))

        p = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        t = threading.Thread(target=print_output, args=(p.stdout, text, None))
        t.daemon = True
        t.start()
        threads.append(t)
        t = threading.Thread(target=print_output, args=(p.stderr, text, os.path.join(log_dir, 'stderr')))
        t.daemon = True
        t.start()
        threads.append(t)

        running.append(p)
        print("[{}] {}".format(" ".join(text), cmd))

    for x in running:
        x.wait()

    for t in threads:
        t.join()


def exec_one(log_dir, cmd, param):
    command = "{} --output {{output}}".format(cmd)

    for name, _val in param:
        command += " --{0} {{{0}}}".format(name)

    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)

    done_files = set()
    done_param = dict()

    for f in tqdm(glob.glob(os.path.join(log_dir, "*.pk"))):
        if f not in done_files:
            done_files.add(f)

            a = to_dict(load_args(f))
            a = tuple((name, a[name] if name in a else None) for name, _vals in param)
            done_param[a] = f

    if param in done_param:
        f = done_param[param]
        def ret(load):
            if load:
                while True:
                    try:
                        r = load_data(f)
                    except EOFError:
                        time.sleep(1)
                    else:
                        return r
        return ret

    for i in count(random.randint(0, 999_999)):
        i = i % 1_000_000
        fn = "{:06d}.pk".format(i)
        fp = os.path.join(log_dir, fn)
        if not os.path.isfile(fp):
            break

    cmd = command.format(output=fp, **dict(param))

    p = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    text = " ".join("{}={}".format(name, val) for name, val in param)

    t1 = threading.Thread(target=print_output, args=(p.stdout, text, None))
    t1.daemon = True
    t1.start()
    t2 = threading.Thread(target=print_output, args=(p.stderr, text, os.path.join(log_dir, 'stderr')))
    t2.daemon = True
    t2.start()

    def ret(load):
        p.wait()
        t1.join()
        t2.join()

        if load:
            return load_data(fp)
    return ret
