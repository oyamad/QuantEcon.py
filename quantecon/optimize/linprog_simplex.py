"""
Contain a linear programming solver routine based on the Simplex Method.

"""
import numpy as np
from numba import jit
from .pivoting import _pivoting, _lex_min_ratio_test


FEA_TOL = 1e-6


def solve_tableau(tableau, basis, max_iter=10**6, phase=2):
    """
    Perform the simplex algorithm on a given tableau in canonical form.

    Used to solve a linear program in the following form:

        maximize:     c @ x

        subject to:   A_ub @ x <= b_ub
                      A_eq @ x == b_eq
                      x >= 0

    where A_ub is of shape (m, n) and A_eq is of shape (k, n). Thus,
    `tableau` is of shape (L+1, n+m+L+1), where L=m+k, and

    * `tableau[np.arange(L), :][:, basis]` must be an identity matrix,
      and
    * the elements of `tableau[:-1, -1]` must be nonnegative.

    Parameters
    ----------
    tableau : ndarray(float, ndim=2)
        ndarray of shape (L+1, n+m+L+1) containing the tableau. Modified
        in place.

    basis : ndarray(int, ndim=1)
        ndarray of shape (L,) containing the basic variables. Modified
        in place.

    max_iter : scalar(int), optional(default=10**6)
        Maximum number of pivoting steps.

    phase : scalar(int), optional(default=2)
        Phase of the simplex algorithm (1 or 2).

    """
    L = tableau.shape[0] - 1

    # Array to store row indices in lex_min_ratio_test
    argmins = np.empty(L, dtype=np.int_)

    success = False
    status = 1
    num_iter = 0

    while num_iter < max_iter:
        pivcol_found, pivcol = _pivot_col(tableau, phase)

        if not pivcol_found:
            success = True
            status = 0
            break

        art_start = tableau.shape[1] - L - 1
        pivrow = _lex_min_ratio_test(tableau[:-1, :], pivcol,
                                     art_start, argmins)
        _pivoting(tableau, pivcol, pivrow)
        basis[pivrow] = pivcol

        num_iter += 1

    return success, status, num_iter


def _pivot_col(tableau, phase):
    L = tableau.shape[0] - 1
    criterion_row_stop = tableau.shape[1] - 1  # Phase 1
    if phase == 2:
        criterion_row_stop -= L

    found = False
    pivcol = -1
    coeff = FEA_TOL
    for j in range(criterion_row_stop):
        if tableau[-1, j] > coeff:
            coeff = tableau[-1, j]
            pivcol = j
            found = True

    return found, pivcol


def get_solution(tableau, basis, x, lambd):
    n, L = x.size, lambd.size
    art_start = tableau.shape[1] - L - 1

    x[:] = 0
    for i in range(L):
    	if basis[i] < n:
	        x[basis[i]] = tableau[i, -1]
    for j in range(L):
        lambd[j] = tableau[-1, art_start+j] * (-1)
    fun = tableau[-1, -1] * (-1)

    return fun
