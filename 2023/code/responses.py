import numpy as np

"""
This file contains the calculation of
(i) a simplified mass constraint and corresponding sensitivities,
(ii) a displacement objective and corresponding sensitivities.

Sensitivities are validated by finite difference.
"""


def constraint(x, alpha=0.5):
    """
    Calculation of mass constraint value and sensitivities.

    :param x: Design variables
    :param alpha: Mass fraction coefficient
    :param scaling: Scaling coefficient
    :return: Constraint value and sensitivities
    """

    return np.sum(x), np.ones_like(x)


def assembly(x, k=1.0):
    """
    Assembly of global stiffness matrix.

    :param x: Design variables
    :param k: Element stiffness Ebt/L
    :return: Stiffness matrix
    """

    # Element stiffness matrix
    Ke = k * np.array([[1, -1], [-1, 1]], dtype=float)

    # Assembly of global stiffness matrix
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

    # Application of boundary conditions and solving the system of equations
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

    # Forward analysis
    u, Ke = analysis(x)

    # Backward analysis
    v = u.copy()

    # if not self-adjoint (meaning dfdu is not force), then
    # v = solve(K, dfdu)
    # note: use same boundary conditions as in forward analysis

    # Sensitivity analysis
    dfdx = np.zeros_like(x, dtype=float)
    for i in range(x.size):
        ue = u[i:i+2]
        ve = v[i:i+2]
        dfdx[i] = - ve @ (Ke @ ue)

    return scaling * u[-1], scaling * dfdx


def finite_difference(fun, x0, h=1e-6):
    """
    Given a response function (fun), finite difference is used to verify the analytical sensitivities.

    :param fun: Response function
    :param x0: Design variables
    :param h: Perturbation magnitude
    :return: Analytical and finite difference sensitivities and their relative error
    """

    # Function value and corresponding sensitivities
    f, dfdx_a = fun(x0)

    # Sensitivities using finite difference
    df = np.zeros_like(x0)
    for i in range(x0.size):
        x = x0.copy()
        x[i] += h
        df[i], _ = fun(x)

    dfdx_fd = (df - f) / h

    # Relative error
    relative_error = (dfdx_fd - dfdx_a) / dfdx_a

    return dfdx_a, dfdx_fd, relative_error


if __name__ == '__main__':
    """
    Check consistency of the sensitivities (objective and constraint) using finite differences.
    """

    x0 = 0.5 * np.ones(4, dtype=float)
    # x0 = np.random.rand(10)

    # Finite difference of constraint function
    dg_ref = np.ones_like(x0) / (0.5 * x0.size)

    dg_a, dg_fd, dg_error = finite_difference(constraint, x0)

    print(f'Constraint \n'
          f'Reference = {dg_ref} \n'
          f'Analytical = {dg_a} \n'
          f'Finite difference = {dg_fd} \n'
          f'Relative error = {dg_error} \n')

    # Finite difference of constraint function
    df_ref = -1 / x0**2
    df_a, df_fd, df_error = finite_difference(objective, x0)

    print(f'Objective \n'
          f'Reference = {df_ref} \n'
          f'Analytical = {df_a} \n'
          f'Finite difference = {df_fd} \n'
          f'Relative error = {df_error} \n')
