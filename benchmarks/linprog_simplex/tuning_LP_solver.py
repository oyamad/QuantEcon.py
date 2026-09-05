"""
Grid search over the pivoting tolerances of `linprog_simplex`.

For every combination of (fea_tol, tol_piv, tol_ratio_diff) in a
cartesian grid, solve each of the selected Netlib test problems and
record the outcome and the wall time of the solve. One row per
(parameter combination, problem) is written to a CSV file with columns

    param_idx, fea_tol, tol_piv, tol_ratio_diff, problem,
    fun, success, status, num_iter, exec_time

Usage:

    python tuning_LP_solver.py [output.csv]

The problems solved are the Netlib problems in linprog_benchmark_files/
without BOUNDS/RANGES sections, excluding QAP15 and STOCFOR3 (too big).
Since the .npz files formulate minimization problems while
`linprog_simplex` maximizes, the objective is negated; the reference
optimal value is thus -problem['obj'].

Note: a full run of the grids used so far takes several hours; a few
problems take 20 minutes each per parameter combination.

"""
import glob
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

from quantecon import cartesian
from quantecon.optimize import linprog_simplex, PivOptions


fea_tol_list = [1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
tol_piv_list = [1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12]
tol_ratio_diff_list = [1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13, 1e-14, 1e-15]

param_set = cartesian((fea_tol_list, tol_piv_list, tol_ratio_diff_list))

max_iter = 10_000

data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'linprog_benchmark_files')
remove_list = ['80BAU3B', 'BORE3D', 'CAPRI', 'CYCLE', 'CZPROB', 'D6CUBE',
               'DFL001', 'ETAMACRO', 'FINNIS', 'FIT1D', 'FIT1P', 'FIT2D',
               'FIT2P', 'GANGES', 'GFRD-PNC', 'GREENBEA', 'GREENBEB',
               'GROW15', 'GROW22', 'GROW7', 'KB2', 'MAROS', 'MODSZK1',
               'PEROLD', 'PILOT', 'PILOT-JA', 'PILOT-WE', 'PILOT4',
               'PILOT87', 'PILOTNOV', 'RECIPE', 'SHELL', 'SIERRA',
               'STAIR', 'STANDATA', 'STANDMPS', 'TUFF', 'VTP-BASE']


def load_problems():
    paths = sorted(glob.glob(os.path.join(data_dir, '*.npz')))
    for name in remove_list:
        paths.remove(os.path.join(data_dir, name + '.npz'))
    problems = {}
    with warnings.catch_warnings():
        # The .npz files were written with Python 2; numpy warns about
        # the extra header parsing needed.
        warnings.simplefilter('ignore', UserWarning)
        for path in paths:
            name = os.path.basename(path)[:-len('.npz')]
            with np.load(path) as data:
                problems[name] = dict(
                    c=-data['c'], A_ub=data['A_ub'], b_ub=data['b_ub'],
                    A_eq=data['A_eq'], b_eq=data['b_eq'],
                )
    return problems


def run(problems, param_set, out_file, max_iter=max_iter):
    rows = []
    columns = ['param_idx', 'fea_tol', 'tol_piv', 'tol_ratio_diff',
               'problem', 'fun', 'success', 'status', 'num_iter',
               'exec_time']
    num_total = len(param_set) * len(problems)
    for i, params in enumerate(param_set):
        piv_options = PivOptions(*params)
        for name, problem in problems.items():
            start = time.perf_counter()
            res = linprog_simplex(problem['c'],
                                  A_ub=problem['A_ub'], b_ub=problem['b_ub'],
                                  A_eq=problem['A_eq'], b_eq=problem['b_eq'],
                                  max_iter=max_iter, piv_options=piv_options)
            exec_time = time.perf_counter() - start
            rows.append((i, *params, name, res.fun, res.success, res.status,
                         res.num_iter, exec_time))
            print(f'\r{len(rows)}/{num_total}', end='', flush=True)
        # Save after every parameter combination so that a partial run
        # is not lost.
        pd.DataFrame(rows, columns=columns).to_csv(out_file, index=False)
    print()
    return pd.DataFrame(rows, columns=columns)


if __name__ == '__main__':
    out_file = sys.argv[1] if len(sys.argv) > 1 else 'results_big.csv'
    problems = load_problems()
    print(f'{len(param_set)} parameter combinations x {len(problems)} '
          f'problems -> {out_file}')
    run(problems, param_set, out_file)
