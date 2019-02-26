from scipy.optimize import newton
import numpy as np


class HyperbolicTangentBVPSolver(object):
    """
    Solve a boundary value problem by assuming a hyperbolic functional form.
    Hence, by solving the Euler equation the only free parameter is the width
    of the interface layer.

    :param list rhs: List of callable objects returning the right handside for
        each equation
    :param np.ndarray mesh_points: Mesh points at which the basis functions are
        evaluated
    :param list boundary_values: Nested list holding the boundary values. 
        Each item containts the lower and upper bound for each variable.
        Example: If we have two variables x1 and x2 satisfying
        0 <= x1 <= 1 and 0.1 <= x2 <= 2.3, the boundary values
        would be [[0, 1], [0.1, 2.3]]
    :param list mass_terms: List with the coefficient in the double derivative
        of the Euler equations.
    """
    def __init__(self, rhs, mesh_points, boundary_values, mass_terms=None):
        self.rhs = rhs
        self.mesh_points = mesh_points
        self.boundary_values = boundary_values
        self.widths = np.ones(len(boundary_values))
        self.mass_terms = mass_terms

        if self.mass_terms is None:
            self.mass_terms = np.ones_like(self.widths)

    def _height(self, var_num):
        bv = self.boundary_values[var_num]
        return bv[1] - bv[0]

    def _basis_func(self, x, var_num):
        height = self._height(var_num)
        w = self.widths[var_num]
        return 0.5*height*(1+np.tanh(x/w)) + self.boundary_values[var_num][0]

    def _basis_func_deriv(self, x, var_num):
        h = self._height(var_num)
        w = self.widths[var_num]
        return 0.5*h/(w*np.cosh(x/w)**2)

    def integrate_rhs(self):
        """Integrate the right hand side."""
        rhs_values = []
        x = self.mesh_points
        profile = np.zeros((len(self.rhs), len(x)))
        for bf in range(profile.shape[0]):
            profile[bf, :] = self._basis_func(x, bf)

        for i in range(len(self.rhs)):
            integrand = self._basis_func(x, i)*self.rhs[i](profile)
            integral = np.trapz(integrand)
            rhs_values.append(integral)
        return rhs_values

    def integrate_lhs(self):
        """Integrate the left hand sides."""
        lhs_values = []
        for i in range(len(self.rhs)):
            integrand = self._basis_func_deriv(self.mesh_points, i)**2
            integral = np.trapz(integrand)
            lhs_values.append(-self.mass_terms[i]*integral)
        return lhs_values

    def solve(self):
        """Find the widths via a Galerkin method"""
        def func(w):
            self.widths = w
            rhs = self.integrate_rhs()
            lhs = self.integrate_lhs()
            return np.array(rhs) - np.array(lhs)

        x0 = np.zeros(len(self.rhs))
        x0 += 0.1*(self.mesh_points[-1] - self.mesh_points[0])
        sol = newton(func, x0)
        self.widths = sol
        return self.widths
