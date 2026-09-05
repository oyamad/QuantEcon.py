# Benchmark / tuning study for `linprog_simplex`

This directory holds a hyperparameter-tuning study for the pivoting
tolerances of `quantecon.optimize.linprog_simplex`, that is, the fields of
`PivOptions`: `fea_tol`, `tol_piv`, `tol_ratio_diff`. It was carried out by
Kyohei Okumura and Quentin Batista in May-June 2020. It is not part of the
`quantecon` package and is not run by the test suite.

## Contents

- `linprog_benchmark_files/`: the Netlib LP Test Problem Set as `.npz`
  files (92 feasible problems, 28 infeasible ones under `infeasible/`).
  Each file contains the arrays `c`, `A_ub`, `b_ub`, `A_eq`, `b_eq`,
  `bounds` and the optimal value `obj` of the *minimization* problem. The
  files are copied from scipy's benchmark suite
  (`benchmarks/linprog_benchmark_files`), where Matt Haberland converted
  the original SIF/MPS files; see `summary.txt` for the problem list and
  the source URLs.
- `tuning_LP_solver.py`: solves each selected problem for every parameter
  combination in a cartesian grid and writes one row per (combination,
  problem). The 54 problems used are those without BOUNDS/RANGES
  sections, excluding QAP15 and STOCFOR3. A full run takes hours.
- `results.csv`, `results_complement.csv`: results of two completed runs
  (125 and 75 parameter combinations, respectively). Columns:
  `param_idx, fea_tol, tol_piv, tol_ratio_diff, problem, fun, success,
  status, num_iter, exec_time`.
- `analysis.ipynb`: analysis of the two runs: success counts per
  combination (as reported by the solver, and by comparing `fun` with the
  Netlib optimum), failures not due to `max_iter`, and runtimes. Its
  criterion is: minimize average runtime among the combinations that
  maximize the number of successes.
- `tuning_LP_solver.ipynb`: the original exploratory notebook.

## Running

```bash
cd benchmarks/linprog_simplex
python tuning_LP_solver.py results_big.csv
```

The grid to run is set at the top of the script. The `.npz` problems are
minimization problems, so the script passes `-c` to `linprog_simplex` and
the reference optimal value is `-obj`.

## Findings so far

Over the combined 200 combinations, the best combination by the above
criterion is `fea_tol=1e-6, tol_piv=1e-5, tol_ratio_diff=1e-11`, with
`tol_piv` on the boundary of the grid explored. The script's grid was
widened accordingly, but that run has not been executed. The current
defaults of `linprog_simplex` (`1e-6, 1e-7, 1e-13`) predate this study.
