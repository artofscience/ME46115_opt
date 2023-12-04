import numpy as np
from scipy.optimize import minimize, Bounds, LinearConstraint
from analysis import analysis
from objective import objective


def model(x):
    u, Ke = analysis(x)
    return objective(x, u, Ke)


if __name__ == "__main__":
    n = 20
    x0 = np.random.rand(n)

    bounds = Bounds(np.zeros_like(x0), np.ones_like(x0))
    linear_constraint = LinearConstraint(np.ones_like(x0), -np.inf, 0.5 * x0.size)

    # In case your constraint is nonlinear, use
    # nonlinear_constraint = NonlinearConstraint(g, -np.inf, 0.5 * x0.size, jac=dg)
    # with g, dg the function calls to constraint value and sensitivities

    opt = {'verbose': 1, 'maxiter': 100, 'disp': True}
    res = minimize(model, x0, method='trust-constr', jac=True,
                   constraints=[linear_constraint], bounds=bounds, tol=1e-6, options= opt)

    print(res.x)
