from scipy.optimize import fsolve
import numpy as np


class GradientCoeffNoExplicitProfile(object):
    def __init__(self, evaluator, boundary,
                 interface_energy, params_vary,
                 num_density, init_grad_coeff=None,
                 apply_energy_correction=False,
                 neg2zero=False):
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
        self._check_dict_consistency()
        self.apply_energy_correction = apply_energy_correction

        if self.grad_coeff is None:
            self.grad_coeff = np.ones(len(self.boundary))
        else:
            self.grad_coeff = np.array(self.grad_coeff)
        self.sqrt_grad_coeff = np.sqrt(self.grad_coeff)
        self.negative2zero = neg2zero

    @property
    def num_variables(self):
        return len(self.boundary)

    def _check_dict_consistency(self):
        """Check the user input."""
        keys = self.boundary.keys()

        pkeys = self.params_vary.keys()
        if any(k not in keys for k in pkeys):
            raise ValueError("Inconsistent dictionary params_vary "
                             "current keys {} expected keys {}"
                             "".format(list(keys), list(pkeys)))

        ikeys = self.interface_energy.keys()
        if any(k not in keys for k in ikeys):
            raise ValueError("Inconsistent dictionary interface_energy "
                             "current keys: {} expected keys: {}"
                             "".format(list(keys), list(ikeys)))

    def calculate_integrals(self):
        integrals = {}
        npoints = 300
        tol = 1E-6
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

            if self.apply_energy_correction:
                slope = (free_energy[-1] - free_energy[0])/(grid[-1] - grid[0])
                intercept = free_energy[0] - slope*grid[0]
                free_energy -= (slope*grid + intercept)
            deriv = self.evaluator.derivative(variables, free_param)**2

            if np.any(free_energy < -tol) and not self.negative2zero:
                minval = np.min(free_energy)
                raise RuntimeError("It appears like forming the interface {} "
                                   "lowers the energy! Minimum surface "
                                   "formation energy {}"
                                   "".format(interface, np.min(free_energy)))

            free_energy[free_energy < 0.0] = 0.0
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
            lhs = np.array(lhs)
            rhs = np.array(rhs)
            return lhs - rhs
        sol, info, ier, mesg = fsolve(func, self.sqrt_grad_coeff,
                                      maxfev=100000, full_output=1)
        print("===================================================")
        print("==        GRADIENT COEFFICIENT SOLVER INFO       ==")
        print("===================================================")
        print(info)
        print("===================================================")
        self.grad_coeff = sol**2
        return self.grad_coeff
