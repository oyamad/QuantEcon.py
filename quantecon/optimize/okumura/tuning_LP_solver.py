# import
from quantecon.optimize import linprog_simplex, PivOptions
from quantecon.gridtools import cartesian

import numpy as np
import pandas as pd
import glob
import time
import progressbar

fea_tol_list = [1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
tol_piv_list = [1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12]
tol_ratio_diff_list = [1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13, 1e-14, 1e-15]

param_set = cartesian((fea_tol_list, tol_piv_list, tol_ratio_diff_list))

param_set = cartesian((fea_tol_list, tol_piv_list, tol_ratio_diff_list))
num_param = param_set.shape[0]

# path
data_dir = './linprog_benchmark_files/'

# full_problem_list
problem_list = glob.glob(data_dir + '*.npz')
problem_list.sort()
remove_list = ['80BAU3B', 'BORE3D', 'CAPRI', 'CYCLE', 'CZPROB', 'D6CUBE',
               'DFL001', 'ETAMACRO', 'FINNIS', 'FIT1D', 'FIT1P', 'FIT2D',
               'FIT2P', 'GANGES', 'GFRD-PNC', 'GREENBEA', 'GREENBEB',
               'GROW15', 'GROW22', 'GROW7', 'KB2', 'MAROS', 'MODSZK1',
               'PEROLD', 'PILOT', 'PILOT-JA', 'PILOT-WE', 'PILOT4',
               'PILOT87', 'PILOTNOV', 'RECIPE', 'SHELL', 'SIERRA',
               'STAIR', 'STANDATA', 'STANDMPS', 'TUFF', 'VTP-BASE'
               ]
for name in remove_list:
    problem_list.remove(data_dir + name + '.npz')


problems = [np.load(problem) for problem in problem_list]
num_problem = len(problems)

cs = tuple(-problem['c'] for problem in problems)
A_eqs = tuple(problem['A_eq'] for problem in problems)
A_ubs = tuple(problem['A_ub'] for problem in problems)
b_eqs = tuple(problem['b_eq'] for problem in problems)
b_ubs = tuple(problem['b_ub'] for problem in problems)

res = pd.DataFrame()
exec_time = pd.DataFrame()

bar = progressbar.ProgressBar(maxval=num_problem*num_param, \
    widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])

i = 0
bar.start()
for param_i in range(num_param):
    piv_options = PivOptions(*param_set[param_i])
    for problem_i in range(num_problem):
        start_time_problem = time.time()
        res.loc[param_i, problem_i] = linprog_simplex(cs[problem_i],
                                                      A_eq=A_eqs[problem_i],
                                                      b_eq=b_eqs[problem_i],
                                                      A_ub=A_ubs[problem_i],
                                                      b_ub=b_ubs[problem_i],
                                                      piv_options=piv_options,
                                                      max_iter=10_000)

        i += 1
        bar.update(i)

        exec_time.loc[param_i, problem_i] = time.time() - start_time_problem

res.to_csv('results_big.csv')
exec_time.to_csv('exec_time_big.csv')

bar.finish()
