from scipy.integrate import solve_bvp
import numpy as np


class CoupledEuler(object):
    def __init__(self, x, rhs, boundary_values, width=None):
        self.x = x
        self.rhs = rhs
        self.boundary_values = boundary_values
        self.width = width

        if self.width is None:
            self.width = 0.1*(x[-1] - x[0])

    def _init_guess(self):
        x0 = 0.5*(self.x[-1] + self.x[0])
        res = []
        x_scaled = (self.x-x0)/self.width
        for b in self.boundary_values:
            height = b[1] - b[0]
            res.append(0.5*height/(self.width*np.cosh(x_scaled)**2))
            res.append(0.5*height*np.tanh(x_scaled) + b[0])
        return np.array(res)

    def solve(self, tol=1E-8):
        initial = self._init_guess()
        n_eq = initial.shape[0]

        def rhs_func(x, y):
            res = np.zeros_like(y)
            for i in range(int(n_eq/2)):
                res[2*i, :] = self.rhs[i](y)
                res[2*i+1, :] = y[2*i, :]
            return res

        def bc(ya, yb):
            res = np.zeros(len(ya))
            for i in range(int(n_eq/2)):
                res[2*i] = ya[2*i+1] - self.boundary_values[i][0]
                res[2*i+1] = yb[2*i+1] - self.boundary_values[i][1]
            return res

        result = solve_bvp(rhs_func, bc, self.x, initial, tol=tol)
        return result["sol"]
