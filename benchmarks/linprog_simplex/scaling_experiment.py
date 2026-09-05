"""
Equilibration experiment: does scaling the rows and columns of the problem
data let the default tol_piv = 1e-7 succeed on the problems it fails?

Usage: python scaling_experiment.py PROBLEM1,PROBLEM2,... OUT.csv
"""
import sys, time, numpy as np, pandas as pd
sys.path.insert(0, '.')
from run_grid import load_problem
from quantecon.optimize import linprog_simplex, PivOptions
ref = pd.read_csv('reference_optima.csv').set_index('problem').obj_ref
names = sys.argv[1].split(',')
def solve(p, po, scale):
    c, A_ub, b_ub, A_eq, b_eq = p['c'], p['A_ub'], p['b_ub'], p['A_eq'], p['b_eq']
    if scale:
        A = np.vstack([A_ub, A_eq]) if A_ub.size and A_eq.size else (A_ub if A_ub.size else A_eq)
        r = np.abs(A).max(axis=1); r[r == 0] = 1          # row scaling
        col = np.abs(A / r[:, None]).max(axis=0); col[col == 0] = 1   # column scaling
        m = A_ub.shape[0]
        A_ub = A_ub / r[:m, None] / col; b_ub = b_ub / r[:m]
        A_eq = A_eq / r[m:, None] / col; b_eq = b_eq / r[m:]
        c = c / col
    t = time.perf_counter()
    res = linprog_simplex(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, max_iter=10_000, piv_options=po)
    return res, time.perf_counter() - t
rows = []
for n in names:
    p = load_problem(n)
    for label, po in [('1e-7', PivOptions(1e-6, 1e-7, 1e-13)), ('1e-5', PivOptions(1e-6, 1e-5, 1e-13))]:
        for scale in [False, True]:
            res, dt = solve(p, po, scale)
            rel = abs(res.fun / ref[n] - 1)
            rows.append((n, label, scale, res.status, res.num_iter, rel < 1e-6, f'{rel:.1e}', round(dt, 2)))
            print(rows[-1], flush=True)
pd.DataFrame(rows, columns=['problem', 'tol_piv', 'scaled', 'status', 'num_iter', 'solved', 'rel_err', 'sec']).to_csv(sys.argv[2], index=False)
