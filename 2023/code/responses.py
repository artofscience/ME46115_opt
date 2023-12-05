import numpy as np

"""
This file contains the calculation of
(i) a simplified mass constraint and corresponding sensitivities,
(ii) a displacement objective and corresponding sensitivities.

Sensitivities are validated by finite difference.
"""


def constraint(x, alpha=0.5, scaling=1.0):
    """
    Calculation of mass constraint value and sensitivities.

    :param x: Design variables
    :param alpha: Mass fraction coefficient
    :param scaling: Scaling coefficient
    :return: Constraint value and sensitivities
    """

    tmp = 1 / (alpha * x.size)

    return scaling * (tmp * np.sum(x) - 1), scaling * tmp * np.ones_like(x)


def assembly(x, k=1.0):
    """
    Assembly of global stiffness matrix.

    :param x: Design variables
    :param k: Element stiffness Ebt/L
    :return: Stiffness matrix
    """
    Ke = k * np.array([[1, -1], [-1, 1]], dtype=float)

    K = np.zeros((x.size + 1, x.size + 1), dtype=float)
    for i in range(x.size):
        K[i:i + 2, :][:, i:i + 2] += x[i] * Ke

    return K, Ke


def solve(K, f):
    """
    Calculates the displacement vector given a stiffness matrix and force vector.

    :param K: Stiffness matrix
    :param f: Force vector
    :return: Displacement vector
    """

    u = np.zeros_like(f)
    u[1::] = np.linalg.solve(K[1::, :][:, 1::], f[1::])
    return u


def analysis(x):
    """
    Calculates the displacement for a tip-loaded bar.

    :param x: Design variables
    :return: Displacement vector
    """

    # Stiffness matrix
    K, Ke = assembly(x, k=1.0)

    # Force
    f = np.zeros(x.size + 1, dtype=float)
    f[-1] = 1.0

    # Displacement
    u = solve(K, f)

    return u, Ke


def objective(x, scaling=1.0):
    """
    Calculation of objective value and sensitivities

    :param x: Design variables
    :param scaling: Scaling coefficient
    :return: Objective value and sensitivities
    """

    # if not self-adjoint (meaning dfdu is not force), then
    # v = solve(K, dfdu)
    # note: use same boundary conditions as in forward analysis

    u, Ke = analysis(x)

    dfdx = np.zeros_like(x, dtype=float)
    for i in range(x.size):
        ue = u[i:i+2]
        dfdx[i] = - ue @ (Ke @ ue)

    return scaling * u[-1], scaling * dfdx


def finite_difference(fun, x0, h=1e-6):
    f, dfdx_a = fun(x0)
    df = np.zeros_like(x0)
    for i in range(x0.size):
        x = x0.copy()
        x[i] += h
        df[i], _ = fun(x)
    dfdx_fd = (df - f) / h
    relative_error = (dfdx_fd - dfdx_a) / dfdx_a
    return dfdx_a, dfdx_fd, relative_error


if __name__ == '__main__':
    """
    Check consistency of the sensitivities (objective and constraint) using finite differences.
    """

    x0 = 0.5 * np.ones(4, dtype=float)
    # x0 = np.random.rand(10)

    dg_ref = np.ones_like(x0) / (0.5 * x0.size)
    dg_a, dg_fd, dg_error = finite_difference(constraint, x0)

    print(f'Constraint \n'
          f'Reference = {dg_ref} \n'
          f'Analytical = {dg_a} \n'
          f'Finite difference = {dg_fd} \n'
          f'Relative error = {dg_error} \n')

    df_ref = -1 / np.ones_like(x0)**2
    df_a, df_fd, df_error = finite_difference(objective, x0)

    print(f'Objective \n'
          f'Reference = {df_ref} \n'
          f'Analytical = {df_a} \n'
          f'Finite difference = {df_fd} \n'
          f'Relative error = {df_error} \n')

