from responses import constraint, objective
import numpy as np
from scipy.optimize import linprog


def tip_loaded_bar(n: int = 2, alpha: float = 0.5, max_iter: int = 50, max_dx: float = 1e-4,
                   ml_init: float = 0.05, ml_incr: float = 1.2, ml_decr: float = 0.5):
    """
    Solves an optimization problem for minimizing displacement of a tip-loaded bar
    given restricted mass.

    :param n: Number of springs / design variables
    :param alpha: Mass fraction relative to maximum mass
    :param max_iter: Maximum number of design iterations
    :param max_dx: Maximum design change
    :param ml_init: Initial move-limit
    :param ml_incr: Move-limit increase parameter
    :param ml_decr: Move-limit decrease parameter
    :return:
    """

    # Set initial design
    x = np.linspace(0.1, 0.9, n)

    # Initialize move-limits
    ml = ml_init * np.ones_like(x)

    # Initialize history
    x_old1, x_old2 = x.copy(), x.copy()

    # Set global bounds
    x_min, x_max = np.zeros_like(x), np.ones_like(x)

    # Objective value of initial design, to be used in scaling
    f0, _ = objective(x)

    # Sequential linear programming loop
    for count in range(max_iter):

        # Objective value and sensitivities
        f, dfdx = objective(x, scaling=1 / f0)

        # Constraint value and sensitivities
        g, dgdx = constraint(x, alpha)

        # Print current status
        print(f'{count:2d}, f: {f:1.3f}, g: {g: 1.3f}, solution: {x}')

        # Set maximum move limit depending on oscillations
        sign = (x - x_old1) * (x_old1 - x_old2)
        ml[sign > 0] = ml_incr * ml[sign > 0]
        ml[sign < 0] = ml_decr * ml[sign < 0]

        # Set lower and upper bounds
        lb = np.maximum(x - ml, x_min)
        ub = np.minimum(x + ml, x_max)

        # Solve linearized minimization problem
        res = linprog(dfdx, A_ub=dgdx[np.newaxis, :],
                      b_ub=[dgdx @ x - g], bounds=np.vstack((lb, ub)).T)

        # Mean absolute design change
        if np.mean(np.abs(res.x - x)) < max_dx:
            break

        # Update history and design variables
        x_old2[:] = x_old1
        x_old1[:] = x
        x[:] = res.x

    return x


if __name__ == "__main__":
    x = tip_loaded_bar(2)

    f, _ = objective(x)
    print(f'\nSolution = {x} \nDisplacement u = {f}')
