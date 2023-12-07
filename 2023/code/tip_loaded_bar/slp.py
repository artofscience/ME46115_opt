from responses import constraint, objective
import numpy as np
from scipy.optimize import linprog

# Set number of springs / design variables
n = 2

# Set mass-fraction
alpha = 0.5  # maximum of alpha material-usage allowed

# Set maximum number of design iterations
iter_max = 50

# Set design change termination criterion
design_change_threshold = 1e-4

# Set move-limit parameters
ml_init, ml_incr, ml_decr = 0.05, 1.2, 0.5

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
for count in range(iter_max):

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
    lb = np.maximum(x - ml, x_min)
    ub = np.minimum(x + ml, x_max)

    # Solve linearized minimization problem
    res = linprog(dfdx, A_ub=dgdx[np.newaxis, :],
                  b_ub=[dgdx @ x - g], bounds=np.vstack((lb, ub)).T)

    # Mean absolute design change
    if np.mean(np.abs(res.x - x)) < design_change_threshold:
        break

    # Update history and design variables
    x_old2[:] = x_old1
    x_old1[:] = x
    x[:] = res.x

f, _ = objective(x)
print(f'\nSolution = {x} \nDisplacement u = {f}')
