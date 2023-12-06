from responses import constraint, objective
import numpy as np
from scipy.optimize import linprog

# Set initial design
x = np.random.rand(2)

# Initialize history
x_old1, x_old2 = x.copy(), x.copy()

# Set move limit parameters
ml = 0.05 * np.ones_like(x)
ml_incr, ml_decr = 1.2, 0.4

# Set global bounds
x_min, x_max = np.zeros_like(x), np.ones_like(x)

# Objective value of initial design
f0, _ = objective(x)

# Set mass-fraction
alpha = 0.5

count = 0  # Initialization of counter
dx = 1.0  # Initialization of design change
bounds = np.zeros((x.size, 2), dtype=float)  # Initialization of variable bounds

# Sequential linear programming loop
while dx > 1e-4 and count < 50:
    count += 1

    # Objective value and sensitivities
    f, dfdx = objective(x, scaling=1/f0)

    # Constraint value and sensitivities
    g, dgdx = constraint(x)

    # Print current status
    print(f'{count:2d}, f: {f:1.3f}, g: {g: 1.3f}, solution: {x}')

    # Set maximum move limit depending on oscillations
    sign = (x - x_old1) * (x_old1 - x_old2)
    ml[sign > 0] = ml_incr * ml[sign > 0]
    ml[sign < 0] = ml_decr * ml[sign < 0]

    # Set lower and upper bounds
    bounds[:, 0] = np.maximum(x - ml, x_min)
    bounds[:, 1] = np.minimum(x + ml, x_max)

    # Solve linearized minimization problem
    res = linprog(dfdx, A_ub=dgdx[np.newaxis, :], b_ub=[dgdx @ x - g], bounds=bounds)

    # Mean absolute design change
    dx = np.mean(np.abs(res.x - x))

    # Update history and design variables
    x_old2[:] = x_old1
    x_old1[:] = x
    x[:] = res.x

f, _ = objective(x)
print(f'Solution = {x}, with displacement u = {f}')












