from cemc.phasefield import CoupledEuler
from scipy.optimize import minimize
from itertools import combinations
import numpy as np


class GradientCoefficient(object):
    def __init__(self, pd_free, mesh_points, number_density,
                 interface_energies, delta_free, boundary_values,
                 params_vary_across_interface, tol=1E-8, width=0.1,
                 max_nodes=1000, prefix=None):
        self.pd_free = pd_free
        self.mesh_points = mesh_points
        self.number_density = number_density
        self.delta_free = delta_free
        self.grad_coeff = np.ones(len(pd_free))
        self.tol = tol
        self.boundary_values = boundary_values
        self.varying_params = params_vary_across_interface
        self.interface_energies = interface_energies
        self.width = width
        self.max_nodes = max_nodes
        self.prefix = None

        # Some user input checks
        if len(pd_free) != len(interface_energies):
            raise ValueError("Number of free energies need to match the "
                             "number of interface energies!")

    def evaluate_one(self, interface, params, ret_sol=False):
        grad_coeff = self.grad_coeff[params]

        # The mass terms in the Euler equation is 2 times
        # the coefficient in front of the gradient terms
        mass_terms = 2*grad_coeff

        rhs = [self.pd_free[x] for x in params]
        b_vals = self.boundary_values[interface]

        euler = CoupledEuler(self.mesh_points, rhs, b_vals,
                             mass_terms=mass_terms, width=self.width)

        sol = euler.solve(tol=self.tol, max_nodes=self.max_nodes)
        deriv = sol.derivative()

        sol_eval = sol(self.mesh_points)
        order_param = sol_eval[1::2, :]
        order_deriv = sol_eval[0::2, :]
        integrand = self.delta_free(order_param)

        for i in range(len(grad_coeff)):
            integrand += grad_coeff[i]*order_deriv[i, :]**2

        result = np.trapz(integrand, x=self.mesh_points)
        integral = result*self.number_density

        if ret_sol:
            return integral, order_param
        return integral

    def find_gradient_coefficients(self, maxiter=10000, eps=0.01):
        def cost_func(grad_coeff):
            sigmas = {}
            self.grad_coeff[:] = grad_coeff
            self.grad_coeff[self.grad_coeff < 1E-4] = 1E-4
            for interface, param in self.varying_params.items():
                res = self.evaluate_one(interface, param)
                sigmas[interface] = res

            rmse = sum((sigmas[k] - self.interface_energies[k])**2
                       for k in sigmas.keys())
            return rmse

        options = {"eps": eps}
        res = minimize(cost_func, self.grad_coeff, method="Nelder-Mead")
        self.grad_coeff = res["x"]
        return self.grad_coeff
