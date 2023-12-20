import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from scipy.optimize import linprog

from responses import constraint, objective_bar, objective_beam


def slp(objective, n: int = 2, alpha: float = 0.5, max_iter: int = 50, max_dx: float = 1e-4,
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
    :return: Optimized design variables
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
        print(f'{count:2d}, f: {f:1.3f}, g: {g: 1.3f}, x: {x}')

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

    # Number of segments / elements
    n = 20

    # Solve bar optimization problem
    x = slp(objective_bar, n)
    f, _ = objective_bar(x)
    print(f'\nSolution bar = {x} \nDisplacement u = {f}\n')

    # Solve beam optimization problem
    y = slp(objective_beam, n)
    f, _ = objective_beam(y)
    print(f'\nSolution beam = {y} \nDisplacement v = {f}\n')

    # Plot design variables over the length of the bar / beam
    fig, ax = plt.subplots(2)
    fig.suptitle('Optimized thickness over length for tip-loaded bar/beam')
    for i in range(x.size):
        ax[0].add_artist(Rectangle((i, -x[i]/2), width=1.0, height=x[i]))
        ax[1].add_artist(Rectangle((i, -y[i]/2), width=1.0, height=y[i]))

    fig.tight_layout(pad=3.0)

    ax[0].title.set_text('Optimized bar')
    ax[0].set_ylim([-0.5, 0.5])
    ax[0].set_xlim([0, n])
    ax[0].plot(0.5 + np.arange(x.size), x/2, 'ro-')

    ax[1].title.set_text('Optimized beam')
    ax[1].set_ylim([-0.5, 0.5])
    ax[1].set_xlim([0, n])
    ax[1].plot(0.5 + np.arange(y.size), y/2, 'ro-')

    plt.show()
