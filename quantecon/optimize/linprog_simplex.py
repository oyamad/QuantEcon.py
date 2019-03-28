"""
Contain a linear programming solver routine based on the Simplex Method.

"""
from collections import namedtuple
import numpy as np
from numba import jit
from .pivoting import _pivoting, _lex_min_ratio_test


FEA_TOL = 1e-6


SimplexResult = namedtuple(
    'SimplexResult', ['x', 'lambd', 'fun', 'success', 'status', 'num_iter']
)


def linprog_simplex(c, A_ub=np.empty((0, 0)), b_ub=np.empty((0,)),
                    A_eq=np.empty((0, 0)), b_eq=np.empty((0,)), max_iter=10**6,
                    tableau=None, basis=None, x=None, lambd=None):
    """
    Solve a linear linear programming in the following form
    by the simplex algorithm:

        maximize:     c @ x

        subject to:   A_ub @ x <= b_ub
                      A_eq @ x == b_eq
                      x >= 0

    where c is of shape (n,1), A_ub is of shape (m, n),
    and A_eq is of shape (k, n).

    Parameters
    ----------
    c : ndarray(float, ndim=1)
        ndarray of shape (n,) containing coefficients of the linear objective
        function to be maximized.

    A_ub : ndarray(float, ndim=2), optional(default=np.empty((0, 0)))
        ndarray of shape (m, n) such that A_eq @ x gives the values of
        the inequality constraints at x.

    b_ub : ndarray(float, ndim=1), optional(default=np.empty((0,)))
        ndarray of shape (m,) representing the RHS of each inequality constraint
        (row) in A_ub.

    A_eq : ndarray(float, ndim=2), optional(default=np.empty((0, 0)))
        ndarray of shape (k, n) such that A_eq @ x gives the values of
        the equality constraints at x.

    b_eq : ndarray(float, ndim=1), optional(default=np.empty((0,)))
        ndarray of shape (k,) representing the RHS of each equality constraint
        (row) in A_eq.

    maxiter : int, optional(default=10**6)
        Maximum number of iteration to perform.

    tableau : ndarray(float, ndim=2), optional(default=None)
        ndarray of shape (m+k+1, n+m+m+k+1) containing the tableau.

    basis : ndarray(int, ndim=1), optional(default=None)
        ndarray of shape (m+k,) containing the basic variables.

    x : ndarray(float, ndim=1), optional(default=None)
        ndarray of shape (n,) containing the independent variable vector.

    lambd : ndarray(float, ndim=1), optional(default=None)
        ndarray of shape (m+n,) containing the dual variable vector.

    Returns
    -------
    SimplexResult : namedtuple
        namedtuple consisting of the fields:
            x : ndarray(float, ndim=1)
                ndarray of shape (n,) containing the independent variable vector
                which optimizes the primal problem.

            lambd : ndarray(float, ndim=1)
                ndarray of shape (m+n,) containing the dual variable vector
                which optimizes the dual problem.

            fun : float
                Value of the objective function.

            success : bool
                Returns True if the algorithm succeeded in finding an optimal
                solution.

            status : int
                An integer representing the exis status of the optimization:
                    0 : Optimization terminated successfully
                    1 : Iteration limit reached
                    2 : Problem appears to be infeasible
                    3 : Problem apperas to be unbounded

            num_iter : int
                The number of iterations performed.

    """
    n, m, k = c.shape[0], A_ub.shape[0], A_eq.shape[0]
    L = m + k

    if tableau is None:
        tableau = np.empty((L+1, n+m+L+1))
    if basis is None:
        basis = np.empty(L, dtype=np.int_)
    if x is None:
        x = np.empty(n)
    if lambd is None:
        lambd = np.empty(L)

    num_iter = 0
    fun = -np.inf

    b_signs = np.empty(L, dtype=np.bool_)
    for i in range(m):
        b_signs[i] = True if b_ub[i] >= 0 else False
    for i in range(k):
        b_signs[m+i] = True if b_eq[i] >= 0 else False

    # Construct initial tableau for Phase 1
    _initialize_tableau(A_ub, b_ub, A_eq, b_eq, tableau, basis)

    # Phase 1
    success, status, num_iter_1 = \
        solve_tableau(tableau, basis, max_iter, skip_aux=False)
    num_iter += num_iter_1
    if not success:  # max_iter exceeded
        return SimplexResult(x, lambd, fun, success, status, num_iter)
    if tableau[-1, -1] > FEA_TOL:  # Infeasible
        success = False
        status = 2
        return SimplexResult(x, lambd, fun, success, status, num_iter)

    # Modify the criterion row for Phase 2
    _set_criterion_row(c, basis, tableau)

    # Phase 2
    success, status, num_iter_2 = \
        solve_tableau(tableau, basis, max_iter-num_iter, skip_aux=True)
    num_iter += num_iter_2
    fun = get_solution(tableau, basis, x, lambd, b_signs)

    return SimplexResult(x, lambd, fun, success, status, num_iter)


def _initialize_tableau(A_ub, b_ub, A_eq, b_eq, tableau, basis):
    """
    Construct initial tableau for Phase 1.

    Suppose that the original linear programming has the following form:

        maximize:     c @ x

        subject to:   A_ub @ x <= b_ub
                      A_eq @ x == b_eq
                      x >= 0

    Let s be a vector of slack variables converting inequality constraint to
    equality constraint so that the problem turns to be the standard form:

        maximize:     c @ x

            subject to:   A_ub @ x + s == b_ub
                          A_eq @ x     == b_eq
                          x, s         >= 0

    Then, let (z1, z2) be a vector of artificial variables used in Phase 1:
    we solve the following LP:

        maximize:     - (1 @ z1 + 1 @ z2)

        subject to:   A_ub @ x + s + z1 == b_ub
                      A_eq @ x  + z2    == b_eq
                      x, s, z1, z2      >= 0

    The tableau needs to be of shape (m+k+1, n+m+m+k+1).

    Parameters
    ----------
    A_ub : see `linprog_simplex`

    b_ub : see `linprog_simplex`

    A_eq : see `linprog_simplex`

    b_eq : see `linprog_simplex`

    tableau : see `solve_tableau`

    basis : see `solve_tableau`

    Returns
    -------
    tableau : see `solve_tableau`
        tableau for phase 1.

    basis : see `solve_tableau`
        initial feasible basis, the artificial variables, for phase 1.

    """
    m, k = A_ub.shape[0], A_eq.shape[0]
    L = m + k
    n = tableau.shape[1] - (m+L+1)

    for i in range(m):
        for j in range(n):
            tableau[i, j] = A_ub[i, j]
    for i in range(k):
        for j in range(n):
            tableau[m+i, j] = A_eq[i, j]

    tableau[:L, n:-1] = 0

    for i in range(m):
        tableau[i, -1] = b_ub[i]
        if tableau[i, -1] < 0:
            for j in range(n):
                tableau[i, j] *= -1
            tableau[i, n+i] = -1
            tableau[i, -1] *= -1
        else:
            tableau[i, n+i] = 1
        tableau[i, n+m+i] = 1
    for i in range(k):
        tableau[m+i, -1] = b_eq[i]
        if tableau[m+i, -1] < 0:
            for j in range(n):
                tableau[m+i, j] *= -1
            tableau[m+i, -1] *= -1
        tableau[m+i, n+m+m+i] = 1

    tableau[-1, :] = 0
    for i in range(L):
        for j in range(n+m):
            tableau[-1, j] += tableau[i, j]
        tableau[-1, -1] += tableau[i, -1]

    for i in range(L):
        basis[i] = n+m+i

    return tableau, basis


def _set_criterion_row(c, basis, tableau):
    """
    Modify the criterion row of the tableau for Phase 2:
    when Phase 1 is completed, the last row of the tableau does not represent
    the relative cost coefficients.

    Parameters
    ----------
    c : see `linprog_simplex`

    basis : see `linprog_simplex`

    tableau : see `linprog_simplex`

    Returns
    -------
    tableau : see `linprog_simplex`
        the last row contains the relative cost coefficients.

    """
    n = c.shape[0]
    L = basis.shape[0]

    for j in range(n):
        tableau[-1, j] = c[j]
    tableau[-1, n:] = 0

    for i in range(L):
        multiplier = tableau[-1, basis[i]]
        for j in range(tableau.shape[1]):
            tableau[-1, j] -= tableau[i, j] * multiplier

    return tableau


def solve_tableau(tableau, basis, max_iter=10**6, skip_aux=True):
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

    skip_aux : bool, optional(default=True)
        Whether to skip the coefficients of the auxiliary (or
        artificial) variables in pivot column selection.

    Returns
    -------
    success : see `linprog_simplex`

    status : see `linprog_simplex`

    num_iter : see `linprog_simplex`

    """
    L = tableau.shape[0] - 1

    # Array to store row indices in lex_min_ratio_test
    argmins = np.empty(L, dtype=np.int_)

    success = False
    status = 1
    num_iter = 0

    while num_iter < max_iter:
        num_iter += 1

        pivcol_found, pivcol = _pivot_col(tableau, skip_aux)

        if not pivcol_found:  # Optimal
            success = True
            status = 0
            break

        aux_start = tableau.shape[1] - L - 1
        pivrow_found, pivrow = _lex_min_ratio_test(tableau[:-1, :], pivcol,
                                                   aux_start, argmins)

        if not pivrow_found:  # Unbounded
            success = False
            status = 3
            break

        _pivoting(tableau, pivcol, pivrow)
        basis[pivrow] = pivcol

    return success, status, num_iter


def _pivot_col(tableau, skip_aux):
    """
    Choose the column containing the pivot element: the column containing
    the maximum positive element in the last row of the tableau is chosen.

    skip_aux should be True in phase 1, and should be False in phase 2.

    Parameters
    ----------
    tableau : see `linprog_simplex`

    skip_aux : see `solve_tableau`

    Returns
    -------
    found : bool
        True iff there is positive element in the last row of the tableau
        (and then pivotting should be conducted)

    pivcol : int
        the index of column containing the pivotting element.
        -1 if there is no need for pivotting.

    """
    L = tableau.shape[0] - 1
    criterion_row_stop = tableau.shape[1] - 1
    if skip_aux:
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


def get_solution(tableau, basis, x, lambd, b_signs):
    """
    Fetch the optimal value of the linear programming.

    Parameters
    ----------
    tableau : see `linprog_simplex`

    basis : see `linprog_simplex`

    x : see `linprog_simplex`

    lambd : see `linprog_simplex`

    b_signs : ndarray(bool, ndim=1)
        ndarray of shape (L,) where L is the number of constraints of the
        original linear programming.
        The i-th element is True iff the i-th element of the vector (b_ub, b_eq)
        is positive.

    Returns
    -------
    fun : float
        The optimal value

    """
    n, L = x.size, lambd.size
    aux_start = tableau.shape[1] - L - 1

    x[:] = 0
    for i in range(L):
        if basis[i] < n:
            x[basis[i]] = tableau[i, -1]
    for j in range(L):
        lambd[j] = tableau[-1, aux_start+j]
        if lambd[j] != 0 and b_signs[j]:
            lambd[j] *= -1
    fun = tableau[-1, -1] * (-1)

    return fun
