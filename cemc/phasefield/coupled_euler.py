from scipy.integrate import solve_bvp
import numpy as np


class CoupledEuler(object):
    """
    Class for solving set of coupled euler equations

    :param array-like x: Mesh points length N
    :param list rhs: List of callable objects with signature
        f(x, y), where x is the independent variable of length
        N and y is the current solution vector of length
        (N, M), where M is the number of first order ODEs

        Even numbered row in y corresponds to the derivative
        of a the variable, while the variable itself is in
        odd numbered row. Hence, if there are 2 variables
        named y0 and y1 the y-matrix would be
        y[0, :] = y0'
        y[1, :] = y0
        y[2, :] = y1'
        y[3, :] = y1

    :param list bounday_values: List of boundary values for each
        variable there is one value at x[0] and one value at x[N-1]
        If there are two variable y0 and y1, this might be
        [[0, 1], [1, 0]], which means that y0[0] = 0,
        y0[N-1] = 1, y1[0] = 1 and y1[N-1] = 0

    :param float width: Initial guess of the width of the
        boundary layer. If None, the initial guess for the solutoin
        is constructed by assuming that the width of the boundary layer
        is 0.1*(x[N-1] - x[0])
    """
    def __init__(self, x, rhs, boundary_values, width=None, mass_terms=None):
        self.x = x
        self.rhs = rhs
        self.boundary_values = boundary_values
        self.width = width
        self.mass_terms = mass_terms

        if self.width is None:
            self.width = 0.1*(x[-1] - x[0])

        if self.mass_terms is None:
            self.mass_terms = np.ones(len(rhs))

    def _init_guess(self):
        x0 = 0.5*(self.x[-1] + self.x[0])
        res = []
        x_scaled = (self.x-x0)/self.width
        for b in self.boundary_values:
            height = b[1] - b[0]
            res.append(0.5*height/(self.width*np.cosh(x_scaled)**2))
            res.append(0.5*height*(1.0 + np.tanh(x_scaled)) + b[0])
        return np.array(res)

    def solve(self, tol=1E-8, max_nodes=1000):
        """
        Solve the boundary value problem

        :param float tol: Tolerance passed to scipy.integrate.solve
        """
        initial = self._init_guess()
        n_eq = initial.shape[0]

        assert n_eq % 2 == 0

        def rhs_func(x, y):
            res = np.zeros_like(y)
            for i in range(int(n_eq/2)):
                res[2*i, :] = self.rhs[i](y)/self.mass_terms[i]
                res[2*i+1, :] = y[2*i, :]
            return res

        def bc(ya, yb):
            res = np.zeros(len(ya))
            for i in range(int(n_eq/2)):
                res[2*i] = ya[2*i+1] - self.boundary_values[i][0]
                res[2*i+1] = yb[2*i+1] - self.boundary_values[i][1]
            return res

        result = solve_bvp(rhs_func, bc, self.x, initial, tol=tol,
                           max_nodes=max_nodes)

        if not result["success"]:
            raise RuntimeError(result["message"])
        return result["sol"]
