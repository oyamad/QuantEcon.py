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

### 2026 rerun on current `main`

- `run_grid.py`: parallel, resumable grid runner (one row per combination
  and problem, with the reference optimum `obj` in the maximization form).
- `reference_optima.csv`: reference optimal values, from Netlib (as stored
  in the `.npz` files) and from scipy's HiGHS; `obj_ref` is the HiGHS value
  where available. Netlib's values for SCRS8 and STOCFOR3 are off by 3e-6.
- `results_2026_defaults.csv`: one pass with the current defaults over all
  54 problems.
- `results_2026.csv`: grid `fea_tol` in {1e-5, 1e-6, 1e-7} x `tol_piv` in
  {1e-3, ..., 1e-10} x `tol_ratio_diff` in {1e-11, 1e-13, 1e-15},
  `max_iter = 10_000`, 51 problems (QAP12, QAP15, STOCFOR3 excluded).
- `results_2026_large.csv`: QAP12, QAP15, STOCFOR3 along `tol_piv` only.
- `results_2026_maxiter.csv`: the hard problems with `max_iter = 100_000`
  at the leading combinations.
- `scaling_experiment.py`, `results_2026_scaling.csv`: does equilibrating
  the problem data let the default `tol_piv` succeed?
- `analysis_2026.ipynb`: analysis of the rerun with the closeness
  criterion (relative error of the objective below 1e-6 with respect to
  `obj_ref`, regardless of the `success` flag).

## Running

```bash
cd benchmarks/linprog_simplex
python tuning_LP_solver.py results_big.csv
```

The grid to run is set at the top of the script. `run_grid.py` runs the
same experiment in parallel and can be resumed; for example

```bash
python run_grid.py results_2026.csv --fea 1e-5,1e-6,1e-7 \
    --piv 1e-3,1e-4,1e-5,1e-6,1e-7,1e-8,1e-9,1e-10 --ratio 1e-11,1e-13,1e-15 \
    --exclude QAP15,QAP12,STOCFOR3 --workers 10
``` The `.npz` problems are
minimization problems, so the script passes `-c` to `linprog_simplex` and
the reference optimal value is `-obj`.

## Findings of the 2020 study

Over the combined 200 combinations, the best combination by the above
criterion is `fea_tol=1e-6, tol_piv=1e-5, tol_ratio_diff=1e-11`, with
`tol_piv` on the boundary of the grid explored. The script's grid was
widened accordingly, but that run has not been executed. The current
defaults of `linprog_simplex` (`1e-6, 1e-7, 1e-13`) predate this study.

## Findings of the 2026 rerun (current `main`, closeness criterion)

Counting a solve as correct only when the objective matches the reference
optimum to a relative error below 1e-6:

- The current defaults `(1e-6, 1e-7, 1e-13)` solve 38 of 54 problems
  (38 of the 51 in the grid). The best combinations solve 43 of 51; four
  tie: `tol_piv` 1e-5 or 1e-4 with `tol_ratio_diff = 1e-11` and any
  `fea_tol`.
- `tol_piv` is the only tolerance with a large effect. The mean number
  solved per combination rises monotonically from 29.7 at 1e-10 to 41.9 at
  1e-4, then falls to 37.8 at 1e-3, where the solver starts to report
  wrong optima as successes (AGG, AGG3, ISRAEL, LOTFI). `fea_tol` is
  irrelevant within {1e-5, 1e-6, 1e-7}.
- Failure categories over all 3,672 grid solves: 71.3% solved; 12.6%
  false "unbounded" (36% of solves at `tol_piv = 1e-10`, 1% at 1e-4);
  11.1% `max_iter` reached; 4.2% success reported with a value off by more
  than 1e-6 (mostly at `tol_piv = 1e-3`); 0.8% false "infeasible".
- `max_iter = 10_000` was the binding constraint for several problems:
  with 100,000 iterations, 25FV47, BNL2, TRUSS and D2Q06C are solved at the
  looser pivot tolerances (14,000 to 54,000 iterations). Combining
  `tol_piv = 1e-4` and the larger budget solves 46 of 51; the remaining
  failures are DEGEN3, QAP8, WOOD1P and, depending on `tol_ratio_diff`,
  SCTAP2 and DEGEN2 (at 1e-13) or TRUSS and FFFFF800 (at 1e-11).
  QAP12, QAP15 and STOCFOR3 do not finish within 10,000 iterations at any
  `tol_piv`.
- `tol_ratio_diff` matters on degenerate problems, in both directions:
  1e-11 solves DEGEN2 in 2,847 iterations where 1e-13 stalls for 100,000,
  while 1e-13 solves TRUSS where 1e-11 ends in a false "unbounded".
- WOOD1P and SCTAP2 end as "unbounded" within a few hundred iterations at
  almost every setting and are the clearest test cases for the ratio test.
- Mechanism: `tol_piv` acts as a guard against small pivots, not as a
  zero detector. Tracing SCSD8 and SCTAP3 shows that a single pivot
  element of size 1e-7 inflates the tableau by 15 to 35 orders of
  magnitude, after which the "unbounded" verdict is taken on garbage.
  Too large a `tol_piv` (1e-3) instead skips rows with genuinely positive
  entries and lets the step leave the feasible region, producing wrong
  optima that pass the optimality test.
- Equilibration (`scaling_experiment.py`, `results_2026_scaling.csv`):
  scaling the rows and columns of the problem data lets the default
  `tol_piv = 1e-7` solve 11 of the 15 hardest problems instead of 6, the
  same as `tol_piv = 1e-5`, including SCTAP2 which no tolerance setting
  solved unscaled. SCSD8 becomes unsolved when scaled.
- Changing the default `tol_piv` to 1e-5 loses no problem on this test
  set, and on the 38 problems solved by both settings the objective values
  agree to 1e-12 with identical pivot paths on 35 to 37 of them.

Suggested next steps: in the refactor of `pivoting.py`, make the ratio
test prefer large pivots among near-minimal ratios (Harris-style two-pass
test) or use tolerances relative to the scale of the tableau, possibly with
equilibration of the input data; moving the default `tol_piv` to 1e-5 is a
safe interim workaround. Keep WOOD1P, SCTAP2, SCSD8 and DEGEN3 as test
cases.
