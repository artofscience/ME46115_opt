import numpy as np

"""
This file contains the calculation of a simplified mass constraint and corresponding sensitivities.
Sensitivities are validated by finite difference.
"""


def constraint(x, alpha=0.5, scaling=1.0):
    """
    Calculation of mass constraint value and sensitivities
    Input: x, design variables 0 < x < 1
    """

    tmp = 1 / (alpha * x.size)

    return scaling * (tmp * np.sum(x) - 1), scaling * tmp * np.ones_like(x)


if __name__ == '__main__':
    "Check consistency of the sensitivities using finite differences"

    h = 1e-6
    n = 10
    alpha = 0.5
    scaling = 10.0

    x0 = np.random.rand(n)
    m, dmdx_a = constraint(x0, alpha, scaling)

    dm = np.zeros_like(x0)
    for i in range(n):
        x = x0.copy()
        x[i] += h
        dm[i], _ = constraint(x, alpha, scaling)

    dmdx_fd = (dm - m) / h

    print(scaling * np.ones_like(x0) / (alpha * x0.size))
    print(dmdx_a)
    print(dmdx_fd)
    print((dmdx_a - dmdx_fd) / dmdx_fd)
