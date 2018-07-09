"""
Contain a linear programming solver routine based on the Simplex Method.

"""
import numpy as np
from numba import jit
from .pivoting import _pivoting, _lex_min_ratio_test


FEA_TOL = 1e-6


def solve_simplex_canonical(tableau, basis, x, lambd, max_iter=10**6):
    """
    Perform the simplex algorithm on a given tableau in canonical form.

    Parameters
    ----------
    tableau : ndarray(float, ndim=2)
        ndarray of shape (m+1, n+m+1). Modified in place.

    basis : ndarray(int, ndim=1)
        ndarray of shape (m,). Has to be initialized with n, ..., n+m-1.
        Modified in place.

    x : ndarray(float, ndim=1)
        ndarray of shape (n,) to store the primal solution. Modified in
        place.

    lambd : ndarray(float, ndim=1)
        ndarray of shape (m,) to store the dual solution. Modified in
        place.

    max_iter : scalar(int), optional(default=10**6)
        Maximum number of pivoting steps.


    """
    m = tableau.shape[0] - 1
    n = tableau.shape[1] - m - 1

    # Array to store row indices in lex_min_ratio_test
    argmins = np.empty(m, dtype=np.int_)

    success = False
    status = 1
    num_iter = 0

    while num_iter < max_iter:
        pivcol_found, pivcol = _pivot_col(tableau)

        if not pivcol_found:
            success = True
            status = 0
            break

        slack_start = n
        pivrow = _lex_min_ratio_test(tableau[:-1, :], pivcol,
                                     slack_start, argmins)
        _pivoting(tableau, pivcol, pivrow)
        basis[pivrow] = pivcol

        num_iter += 1

    fun = get_solution(tableau, basis, x, lambd)
    return fun, success, status, num_iter


def _pivot_col(tableau):
    L = tableau.shape[1] - 1
    found = False
    pivcol = -1
    coeff = FEA_TOL
    for j in range(L):
        if tableau[-1, j] > coeff:
            coeff = tableau[-1, j]
            pivcol = j
            found = True

    return found, pivcol


def get_solution(tableau, basis, x, lambd):
    n, m = x.size, lambd.size
    x[:] = 0
    for i in range(m):
    	if basis[i] < n:
	        x[basis[i]] = tableau[i, -1]
    for j in range(m):
        lambd[j] = tableau[-1, n+j] * (-1)
    fun = tableau[-1, -1] * (-1)

    return fun
