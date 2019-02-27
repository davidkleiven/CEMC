from scipy.optimize import newton
import numpy as np


class InterfaceWidthError(Exception):
    pass


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
    def __init__(self, rhs, mesh_points, boundary_values, mass_terms=None, 
                 width=0.1, all_same_width=False):
        self.rhs = rhs
        self.mesh_points = mesh_points
        self.boundary_values = boundary_values
        self.widths = width*np.ones(len(boundary_values))
        self.mass_terms = mass_terms
        self.all_same_width = all_same_width

        if self.mass_terms is None:
            self.mass_terms = np.ones_like(self.widths)

    @property
    def num_eq(self):
        return len(self.boundary_values)

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
        profile = np.zeros((2*self.num_eq, len(x)))
        for bf in range(self.num_eq):
            profile[2*bf, :] = self._basis_func_deriv(x, bf)
            profile[2*bf+1, :] = self._basis_func(x, bf)

        for i in range(self.num_eq):
            integrand = self._basis_func(x, i)*self.rhs[i](profile)
            integral = np.trapz(integrand)
            rhs_values.append(integral)
        return rhs_values

    def integrate_lhs(self):
        """Integrate the left hand sides."""
        lhs_values = []
        for i in range(self.num_eq):
            integrand = self._basis_func_deriv(self.mesh_points, i)**2
            integral = np.trapz(integrand)
            lhs_values.append(-self.mass_terms[i]*integral)
        return lhs_values

    def solve(self):
        """Find the widths via a Galerkin method"""
        def func(w):
            self.check_widths(w)
            self.widths = w
            rhs = self.integrate_rhs()
            lhs = self.integrate_lhs()
            return np.array(rhs) - np.array(lhs)

        rng = self.mesh_points[-1] - self.mesh_points[0]
        widths = np.linspace(0.01*rng, 0.4*rng, 100)
        self.find_init_widths(widths.tolist())
        x0 = self.widths
        sol = newton(func, x0, maxiter=500)
        self.widths = sol
        return self.construct_solution()

    def check_widths(self, w):
        rng = self.mesh_points[-1] - self.mesh_points[0]
        if np.max(w) > 0.8*rng:
            msg = "The interface width exceeds "
            msg += "80 percent of the overall domain width"
            msg += "\nWidths: {}".format(w)
            msg += "\nDomain size: {}".format(rng)
            raise InterfaceWidthError(msg)

    def construct_solution(self):
        """Construct the solution array."""
        sol = np.zeros((2*self.num_eq, len(self.mesh_points)))
        x = self.mesh_points
        for i in range(self.num_eq):
            sol[2*i, :] = self._basis_func_deriv(x, i)
            sol[2*i+1, :] = self._basis_func(x, i)
        return sol

    def find_init_widths(self, widths, show=False):
        all_rhs = []
        all_lhs = []
        for w in widths:
            self.widths[:] = w
            lhs = self.integrate_lhs()
            rhs = self.integrate_rhs()
            all_lhs.append(lhs)
            all_rhs.append(rhs)

        diff = np.array(all_lhs[0]) - np.array(all_rhs[0])
        min_indx = np.argmin(np.abs(diff))
        self.widths[:] = widths[min_indx]

        if show:
            from matplotlib import pyplot as plt
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.plot(widths, all_lhs, label="LHS", marker="o")
            ax.plot(widths, all_rhs, label="RHS", marker="o")
            ax.legend()
            plt.show()
