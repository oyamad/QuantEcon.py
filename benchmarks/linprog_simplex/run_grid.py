"""
Parallel grid search over the pivoting tolerances of `linprog_simplex`.

Solves each selected Netlib problem for every (fea_tol, tol_piv,
tol_ratio_diff) in a cartesian grid, in parallel over (combination,
problem) tasks, and appends one row per task to a CSV file with columns

    param_idx, fea_tol, tol_piv, tol_ratio_diff, problem,
    fun, obj, success, status, num_iter, exec_time

where `obj` is the Netlib optimal value (negated, for maximization).
The run is resumable: tasks already present in the output are skipped.

Usage:
    python run_grid.py OUT.csv [--fea a,b,...] [--piv a,b,...]
        [--ratio a,b,...] [--max-iter N] [--workers W]
        [--exclude NAME,...] [--only NAME,...]
"""
import argparse
import csv
import glob
import os
import time
import warnings
from multiprocessing import Pool

import numpy as np

from quantecon import cartesian
from quantecon.optimize import linprog_simplex, PivOptions


DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'linprog_benchmark_files')
# Problems with BOUNDS/RANGES (not supported), plus QAP15 and STOCFOR3
# (too big), as in the 2020 study.
REMOVE_LIST = ['80BAU3B', 'BORE3D', 'CAPRI', 'CYCLE', 'CZPROB', 'D6CUBE',
               'DFL001', 'ETAMACRO', 'FINNIS', 'FIT1D', 'FIT1P', 'FIT2D',
               'FIT2P', 'GANGES', 'GFRD-PNC', 'GREENBEA', 'GREENBEB',
               'GROW15', 'GROW22', 'GROW7', 'KB2', 'MAROS', 'MODSZK1',
               'PEROLD', 'PILOT', 'PILOT-JA', 'PILOT-WE', 'PILOT4',
               'PILOT87', 'PILOTNOV', 'RECIPE', 'SHELL', 'SIERRA',
               'STAIR', 'STANDATA', 'STANDMPS', 'TUFF', 'VTP-BASE']
COLUMNS = ['param_idx', 'fea_tol', 'tol_piv', 'tol_ratio_diff', 'problem',
           'fun', 'obj', 'success', 'status', 'num_iter', 'exec_time']


def problem_names(exclude=(), only=None):
    names = sorted(os.path.basename(p)[:-4]
                   for p in glob.glob(os.path.join(DATA_DIR, '*.npz')))
    names = [n for n in names if n not in REMOVE_LIST and n not in exclude]
    if only:
        names = [n for n in names if n in only]
    return names


def load_problem(name):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)  # Python 2 npz header
        with np.load(os.path.join(DATA_DIR, name + '.npz')) as d:
            return dict(c=-d['c'], A_ub=d['A_ub'], b_ub=d['b_ub'],
                        A_eq=d['A_eq'], b_eq=d['b_eq'], obj=-float(d['obj']))


def solve(task):
    param_idx, params, name, max_iter = task
    p = load_problem(name)
    start = time.perf_counter()
    res = linprog_simplex(p['c'], A_ub=p['A_ub'], b_ub=p['b_ub'],
                          A_eq=p['A_eq'], b_eq=p['b_eq'], max_iter=max_iter,
                          piv_options=PivOptions(*params))
    exec_time = time.perf_counter() - start
    return (param_idx, *params, name, res.fun, p['obj'], res.success,
            res.status, res.num_iter, exec_time)


def floats(s):
    return [float(x) for x in s.split(',')]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--fea', type=floats, default=[1e-6])
    ap.add_argument('--piv', type=floats, default=[1e-7])
    ap.add_argument('--ratio', type=floats, default=[1e-13])
    ap.add_argument('--max-iter', type=int, default=10_000)
    ap.add_argument('--workers', type=int, default=max(1, os.cpu_count() - 2))
    ap.add_argument('--exclude', type=lambda s: s.split(','), default=[])
    ap.add_argument('--only', type=lambda s: s.split(','), default=None)
    args = ap.parse_args()

    grid = cartesian((args.fea, args.piv, args.ratio))
    names = problem_names(args.exclude, args.only)

    done = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            for row in csv.DictReader(f):
                done.add((int(row['param_idx']), row['problem']))
    tasks = [(i, tuple(params), n, args.max_iter)
             for i, params in enumerate(grid) for n in names
             if (i, n) not in done]
    # Largest problems first for load balancing.
    size = {n: os.path.getsize(os.path.join(DATA_DIR, n + '.npz')) for n in names}
    tasks.sort(key=lambda t: -size[t[2]])
    print(f'{len(grid)} combinations x {len(names)} problems; '
          f'{len(tasks)} tasks to run ({len(done)} done) with '
          f'{args.workers} workers -> {args.out}', flush=True)

    new_file = not os.path.exists(args.out)
    start = time.time()
    with open(args.out, 'a', newline='') as f, Pool(args.workers) as pool:
        w = csv.writer(f)
        if new_file:
            w.writerow(COLUMNS)
        for k, row in enumerate(pool.imap_unordered(solve, tasks), 1):
            w.writerow(row)
            f.flush()
            if k % 50 == 0 or k == len(tasks):
                el = time.time() - start
                print(f'{k}/{len(tasks)} done, {el/60:.1f} min elapsed, '
                      f'eta {el/k*(len(tasks)-k)/60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
