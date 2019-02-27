from scipy.optimize import minimize, newton
from scipy.optimize import LinearConstraint
import numpy as np


class GradientCoeffNoExplicitProfile(object):
    def __init__(self, evaluator, boundary,
                 interface_energy, params_vary,
                 num_density, init_grad_coeff=None):
        from cemc.phasefield import GradCoeffEvaluator

        if not isinstance(evaluator, GradCoeffEvaluator):
            raise TypeError("Evaluator has to be an instance of "
                            "GradCoeffEvaluator!")

        self.evaluator = evaluator
        self.boundary = boundary
        self.grad_coeff = init_grad_coeff
        self.params_vary = params_vary
        self.num_density = num_density
        self.interface_energy = interface_energy

        if self.grad_coeff is None:
            self.grad_coeff = np.ones(len(self.boundary))
        self.sqrt_grad_coeff = np.sqrt(self.grad_coeff)

    @property
    def num_variables(self):
        return len(self.boundary)

    def calculate_integrals(self):
        integrals = {}
        npoints = 300
        for interface in self.boundary.keys():

            # Take the first varying parameter as the integration
            # variable
            free_param = self.params_vary[interface][0]

            b = self.boundary[interface]
            grid = np.linspace(b[free_param][0], b[free_param][1], npoints)
            variables = []
            varying_params = []
            for i in range(self.num_variables):
                if i == free_param:
                    variables.append(grid)
                elif abs(b[i][0] - b[i][1]) < 1E-6:
                    variables.append(b[i][0])
                else:
                    variables.append(None)
                    varying_params.append(i)

            free_energy = self.evaluator.evaluate(variables, free_param)
            deriv = self.evaluator.deriv(variables, free_param)**2

            if np.any(free_energy < 0.0):
                raise RuntimeError("It appears like forming an interface "
                                   "lowers the energy!")

            integrand = np.sqrt(free_energy)
            grad_terms = deriv.T.dot(self.grad_coeff).T

            dx = grid[1] - grid[0]
            integral = np.trapz(integrand*np.sqrt(grad_terms), dx=dx)
            integrals[interface] = 2*self.num_density*integral
        return integrals

    def solve(self):
        def func(sqrt_grad_coeff):
            self.grad_coeff = sqrt_grad_coeff**2
            integrals = self.calculate_integrals()
            rhs = []
            lhs = []
            for interface in integrals.keys():
                rhs.append(integrals[interface])
                lhs.append(self.interface_energy[interface])
            return np.array(lhs) - np.array(rhs)

        sol = newton(func, self.sqrt_grad_coeff)
        self.grad_coeff = sol**2
        print(self.grad_coeff)
        return self.grad_coeff