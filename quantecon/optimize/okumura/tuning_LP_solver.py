# import
from quantecon.optimize import linprog_simplex, PivOptions
#from quantecon.gridtools import cartesian

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from collections import namedtuple
from itertools import product
import glob
from multiprocessing import Pool, cpu_count


def main(row_i):
    param_i = row_i // num_param
    problem_i = row_i % num_param

    piv_options = PivOptions(*param_set[param_i])

    desired_fun = desired_funs[problem_i]
    res = linprog_simplex(cs[problem_i],
                          A_eq=A_eqs[problem_i],
                          b_eq=A_ubs[problem_i],
                          A_ub=b_eqs[problem_i],
                          b_ub=b_ubs[problem_i],
                          piv_options=piv_options,
                          max_iter=10_000)

    result_dict = {'param': str(param), 'problem': problem_nb,
                   'res.fun': res.fun, 'desired_fun': -desired_fun,
                   'success?': np.isclose(res.fun, -desired_fun),
                   'status': res.status}
    return result_dict


if __name__ == '__main__':
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

    # fea_tol_list = [1e-5, 5e-6, 1e-6, 5e-7, 1e-7]
    # tol_piv_list = [1e-7, 5e-8, 1e-8, 5e-9, 1e-9]
    # tol_ratio_diff_list = [1e-15, 5e-15, 1e-14, 5e-14, 1e-13]
    fea_tol_list = [1e-5, 1e-6]
    tol_piv_list = [1e-7, 1e-8]
    tol_ratio_diff_list = [1e-15, 1e-14]

    param_set = product(fea_tol_list, tol_piv_list, tol_ratio_diff_list)

#     problem_list = ['./linprog_benchmark_files/AFIRO.npz',
#                     './linprog_benchmark_files/AGG.npz']

    num_problem = len(problem_list)
    num_param =\
        len(fea_tol_list) * len(tol_piv_list) * len(tol_ratio_diff_list)
    num_row = num_problem * num_param
    column_list =\
        ['param', 'problem', 'res.fun', 'desired_fun', 'success?', 'status']
    num_column = len(column_list)
    df = pd.DataFrame(np.zeros((num_row, num_column)))
    df.columns = column_list
    row_index = 0

    problems = [np.load(problem) for problem in problem_list]

    cs = tuple(-problem['c'] for problem in problems)
    A_eqs = tuple(problem['A_eq'] for problem in problems)
    A_ubs = tuple(problem['A_ub'] for problem in problems)
    b_eqs = tuple(problem['b_eq'] for problem in problems)
    b_ubs = tuple(problem['b_ub'] for problem in problems)
    desired_funs = tuple(problem['obj'] for problem in problems)

    # parallelization
    n_cores = cpu_count() // 2
    with Pool(n_cores, maxtasksperchild=1000) as p:
        result_dicts = p.imap(main, range(num_row))

    # convertion to pandas dataframe
    for row_index, result_dict in enumerate(result_dicts):
        df['param'][row_index] = result_dict['param']
        df['problem'][row_index] = result_dict['problem']
        df['res.fun'][row_index] = result_dict['res.fun']
        df['desired_fun'][row_index] = result_dict['desired_fun']
        df['success?'][row_index] = result_dict['success?']
        df['status'][row_index] = result_dict['status']

    print(df)
    df.to_csv('results.csv')

    # visualization
    result_matrix = np.zeros((num_param, num_problem))

    result_column = df['success?']
    for i in range(num_param):
        result_matrix[i] = result_column[i:i+num_problem]

    plt.spy(result_matrix, markersize=3)
    plt.savefig('result_table.png')
    plt.show()
