from cemc.phasefield import CoupledEuler
from scipy.optimize import minimize
from itertools import combinations
import numpy as np


class GradientCoefficient(object):
    """
    Class for finding gradient coefficient by numerically solving
    the Euler equations.

    :param list pd_free: List of callable objects that returns
        the partial derivative of the free energy with respect
        to each of the order parameters. The signature of the
        callable method should be fun(y), where y is an
        NxM dimensional array. N is the number of order parameters
        and M is the number of mesh points
    :param ndarray mesh_points: 1D numby array of length M with the
        mesh points at which the free energy density and the order
        paremeters should be evaluated.
    :param float number_density: Number density of the system
    :param dict interface_energies: Dictionary describing the interface
        energy between two phases. The key is a tuple of integers indicating
        which phases that are considered, and the value is the interface
        energy. Example: If we have three phases this dictionary may look like
        {(0, 1): 0.2, (0, 2): 1.3, (1, 2): 0.01}
    :param callable delta_free: Callable object that can return the free energy
        density. The signature is delta_free(y), where y is an NxM array,
        where N is the number order parameters and M is the number of mesh
        points.
    :param dict boundary_values: Dictionary with boundary values for
        each of the order parameters in question, when crossing
        a particular interface. Example: If we have three phases, but
        on each interface type there are only two order parameters
        that vary. Then this dictionary could look like this
        {(0, 1): [[0, 1], [1, 0]], (0, 2): [[0, 1], [0, 1]],
         (1, 2): [[1, 0], [1, 0]]}
    :param dict params_vary_across_interface: Dictionary describing which
        order parameters that vary across a given interface. For the example
        above this could look like: {(0, 1): [0, 1], (0, 2): [0, 2], 
            (1, 2): [1, 2]}
    :param float tol: Tolerance passed to the CoupledEuler solver
    :param float width: Initial guess for the interface width passed to 
        the CoupledEuler solver
    :param int max_nodes: Maximum number of collocation points (used by 
        scipy.integrate.solve_bvp)
    """
    def __init__(self, rhs_builder, mesh_points, number_density,
                 interface_energies, params_vary_across_interface,
                 tol=1E-8, width=0.1, max_nodes=1000, solver="collocation",
                 init_grad=1.0):
        from cemc.phasefield import GradientCoefficientRhsBuilder
        if not isinstance(rhs_builder, GradientCoefficientRhsBuilder):
            raise TypeError("rhs_builder has to be derived from GradientCoefficientRhsBuilder!")

        self.rhs_builder = rhs_builder
        self.mesh_points = mesh_points
        self.number_density = number_density
        self.grad_coeff = np.ones(len(interface_energies))*init_grad
        self.tol = tol
        self.varying_params = params_vary_across_interface
        self.interface_energies = interface_energies
        self.width = width
        self.max_nodes = max_nodes
        self.prefix = None

        allowed_solvers = ["collocation", "hypertangent"]
        if solver not in allowed_solvers:
            raise ValueError("Solver has to be one of {}"
                             "".format(allowed_solvers))
        self.solver = solver

    def evaluate_one(self, interface, params, ret_sol=False):
        """
        Evaluate one surface profile

        :param tuple interface: Tuple describing the two interfaces in
            question. Example: (0, 1), (0, 2) etc.
        :param list params: List of indices to the order parameters that
            vary across the interface
        :param bool ret_sol: If True the interface profile is returned
            together with the integral value. Otherwise only the integral
            is returned.
        :return: Either float with the estimate interfacial energy or
            the estimated interface energy and a numpy matrix with the
            calculated surface profile
        :rtype: float or float, np.ndarray
        """
        grad_coeff = self.grad_coeff[params]

        # The mass terms in the Euler equation is 2 times
        # the coefficient in front of the gradient terms
        mass_terms = 2*grad_coeff

        rhs, b_vals = self.rhs_builder.construct_rhs_and_boundary(interface)

        euler = CoupledEuler(self.mesh_points, rhs, b_vals,
                             mass_terms=mass_terms, width=self.width)

        sol = euler.solve(tol=self.tol, max_nodes=self.max_nodes,
                          solver=self.solver)

        order_param = sol[1::2, :]
        order_deriv = sol[0::2, :]
        rhs_func = self.rhs_builder.get_projected(interface)
        integrand = rhs_func(sol)
        integrand[integrand < 0.0] = 0.0
        if np.any(integrand < 0.0):
            raise ValueError("It looks like forming an interface "
                             "decrease the energy!")

        dx = self.mesh_points[1] - self.mesh_points[0]
        integral = self.number_density*np.trapz(integrand, dx=dx)
        order_deriv_int = self.number_density*np.trapz(order_deriv**2, axis=1,
                                                       dx=dx)
        return integral, order_deriv_int

    def find_gradient_coefficients(self):
        N = len(self.interface_energies)

        converged = False
        while not converged:
            matrix = np.zeros((N, N))
            rhs = np.zeros(N)
            row = 0
            for interface, param in self.varying_params.items():
                integral, gradient_terms = self.evaluate_one(interface, param)
                rhs[row] = self.interface_energies[interface] - integral
                matrix[row, param] = gradient_terms
                row += 1

            prev_grad_coeff = self.grad_coeff.copy()
            self.grad_coeff = np.linalg.solve(matrix, rhs)
            if np.any(self.grad_coeff < 0.0):
                raise RuntimeError("Some gradient coefficients are negative. "
                                   "Try to change the initial guess...")
            converged = np.max(np.abs(prev_grad_coeff-self.grad_coeff)) < 1E-6
        return self.grad_coeff