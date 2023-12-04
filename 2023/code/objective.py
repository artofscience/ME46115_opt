import numpy as np
from model import analysis, solve

"""
This file contains the calculation of the objective.
Sensitivities are validated by finite difference.
"""


def objective(x, scaling=1.0):
    """
    Calculation of objective value and sensitivities

    :param x: Design variables
    :return:
    """

    u, Ke = analysis(x)

    # if not self-adjoint (meaning dfdu is not force), then
    # v = solve(K, dfdu)
    # note: use same boundary conditions as in forward analysis

    dfdx = np.zeros_like(x, dtype=float)
    for i in range(x.size):
        ue = u[i:i+2]
        dfdx[i] = - ue @ (Ke @ ue)

    return scaling * u[-1], scaling * dfdx


if __name__ == '__main__':
    "Check consistency of the sensitivities using finite differences"

    h = 1e-6  # somewhere between e-3 and e-13
    n = 2
    scaling = 10.0

    x0 = 1.0 * np.ones(n)
    f, dfdx_a = objective(x0, scaling)

    df = np.zeros_like(x0)
    for i in range(n):
        x = x0.copy()
        x[i] += h
        df[i], _ = objective(x, scaling)

    dfdx_fd = (df - f) / h

    print(-scaling / x0**2)
    print(dfdx_a)
    print(dfdx_fd)
    print((dfdx_a - dfdx_fd) / dfdx_fd)
