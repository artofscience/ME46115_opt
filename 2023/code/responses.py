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

    g = np.sum(x) / (alpha * x.size) - 1
    dgdx = np.ones_like(x) / (alpha * x.size)

    return scaling * g, scaling * dgdx


def assembly_bar(x, E=1.0, b=1.0, t=1.0, L=1.0):
    """
    Assembly of global stiffness matrix.

    :param x: Design variables
    :param E: Young's modulus
    :param b: Beam width
    :param t: Maximum beam thickness
    :param L: Total beam length
    :return: Stiffness matrix
    """
    # Element length
    l = L / x.size

    # Element stiffness matrix
    k = E * b * t / l
    Ke = k * np.array([[1, -1], [-1, 1]], dtype=float)

    # Assembly of global stiffness matrix
    K = np.zeros((x.size + 1, x.size + 1), dtype=float)  # number of dof = number of elements + 1
    for i in range(x.size):
        K[i:i + 2, :][:, i:i + 2] += x[i] * Ke

    return K, Ke


def objective_bar(x, scaling=1.0):
    """
    Calculation of objective value and sensitivities

    :param x: Design variables
    :param scaling: Scaling coefficient
    :return: Objective value and sensitivities
    """

    # Forward analysis
    K, Ke = assembly_bar(x)

    # Force
    p = np.zeros(x.size + 1, dtype=float)
    p[-1] = 1.0

    # Application of boundary conditions and solving the system of equations
    u = np.zeros_like(p)
    u[1::] = np.linalg.solve(K[1::, :][:, 1::], p[1::])

    # Objective value
    f = p @ u  # or equivalently u[-1]

    # Backward analysis
    # v = solve(K, dfdu), but we know v = u, since dfdu = f

    # Sensitivity analysis
    dfdx = np.zeros_like(x, dtype=float)
    for i in range(x.size):
        ue = u[i:i+2]
        dfdx[i] = - ue @ (Ke @ ue)

    return scaling * f, scaling * dfdx


def assembly_beam(x, E=1.0, b=1.0, t=1.0, L=1.0):
    """
    Assembly of global stiffness matrix.

    :param x: Design variables
    :param E: Young's modulus
    :param b: Beam width
    :param t: Maximum beam thickness
    :param L: Total beam length
    :return: Stiffness matrix
    """

    # Element length
    l = L / x.size

    # Element stiffness matrix
    k = (E * b * t**3) / (12 * l**3)
    Ke = k * np.array([[12, 6*l, -12, 6*l],
                       [6*l, 4*l**2, -6*l, 2*l**2],
                       [-12, -6*l, 12, -6*l],
                       [6*l, 2*l**2, -6*l, 4*l**2]], dtype=float)

    # Assembly of global stiffness matrix
    n = 2 * (x.size + 1)
    K = np.zeros((n, n), dtype=float)  # number of dof = 2 * number of nodes
    for i in range(x.size):
        K[2*i:2*i + 4, :][:, 2*i:2*i + 4] += x[i]**3 * Ke

    return K, Ke


def objective_beam(x, scaling=1.0):
    """
    Calculation of objective value and sensitivities

    :param x: Design variables
    :param scaling: Scaling coefficient
    :return: Objective value and sensitivities
    """

    # Forward analysis
    K, Ke = assembly_beam(x)

    # Force
    p = np.zeros(2 * (x.size + 1), dtype=float)
    p[-2] = 1.0

    # Application of boundary conditions and solving the system of equations
    u = np.zeros_like(p)
    u[2::] = np.linalg.solve(K[2::, :][:, 2::], p[2::])

    # Objective value
    f = p @ u  # or equivalently u[-2]

    # Backward analysis
    # v = solve(K, dfdu), but we know v = u, since dfdu = f

    # Sensitivity analysis
    dfdx = np.zeros_like(x, dtype=float)
    for i in range(x.size):
        ue = u[2*i:2*i+4]
        dfdx[i] = - 3 * x[i]**2 * ue @ (Ke @ ue)

    return scaling * f, scaling * dfdx


def finite_difference(fun, x0, h=1e-6):
    """
    Given a response function (fun), finite difference is used to verify the analytical sensitivities.

    :param fun: Response function
    :param x0: Design variables
    :param h: Perturbation magnitude
    :return: Analytical and finite difference sensitivities and their relative error
    """

    # Function value and corresponding sensitivities
    f0, _ = fun(x0)

    # Sensitivities using finite difference
    df = np.zeros_like(x0)
    for i in range(x0.size):
        x = x0.copy()
        x[i] += h
        df[i], _ = fun(x)

    return (df - f0) / h


if __name__ == '__main__':
    """
    Check consistency of the sensitivities (objective and constraint) using finite differences.
    """

    alpha = 0.5
    x0 = np.random.rand(3)

    # Finite difference of constraint function
    dg_ref = np.ones_like(x0) / (alpha * x0.size)
    _, dg_a = constraint(x0)
    dg_fd = finite_difference(constraint, x0)

    dg_error = (dg_fd - dg_a) / dg_a

    print(f'Constraint \n'
          f'Reference = {dg_ref} \n'
          f'Analytical = {dg_a} \n'
          f'Finite difference = {dg_fd} \n'
          f'Relative error = {dg_error} \n')

    # Finite difference of objective function
    df_ref = -1 / x0**2
    _, df_a = objective_bar(x0)
    df_fd = finite_difference(objective_bar, x0)

    df_error = (df_fd - df_a) / df_a

    print(f'Objective BAR \n'
          f'Reference = {df_ref} \n'
          f'Analytical = {df_a} \n'
          f'Finite difference = {df_fd} \n'
          f'Relative error = {df_error} \n')

    # Finite difference of objective function
    _, df_a = objective_beam(x0)
    df_fd = finite_difference(objective_beam, x0)

    df_error = (df_fd - df_a) / df_a

    print(f'Objective BEAM \n'
          f'Analytical = {df_a} \n'
          f'Finite difference = {df_fd} \n'
          f'Relative error = {df_error} \n')
