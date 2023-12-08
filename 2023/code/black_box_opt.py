import numpy as np
from scipy.optimize import minimize, Bounds, LinearConstraint
from responses import objective_beam

# Set initial design
x0 = np.linspace(0.1, 0.9, 3)

# Set global variable bounds
bounds = Bounds(np.zeros_like(x0), np.ones_like(x0))

# Setup linear constraint
linear_constraint = LinearConstraint(np.ones_like(x0), -np.inf, 0.5 * x0.size)

# In case your constraint is nonlinear, use
# nonlinear_constraint = NonlinearConstraint(g, -np.inf, 0.5 * x0.size, jac=dg)
# with g, dg the function calls to constraint value and sensitivities

# Set some options for the optimizer
opt = {'verbose': 2, 'maxiter': 100, 'disp': True}

# Solve the nonlinear minimization problem
res = minimize(objective_beam, x0, jac=True, method='trust-constr',
               constraints=linear_constraint, bounds=bounds,
               tol=1e-6, options=opt)

print(f'Solution = {res.x}')
