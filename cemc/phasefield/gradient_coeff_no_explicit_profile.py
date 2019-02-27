from scipy.optimize import newton


class GradientCoeffNoExplicitProfile(object):
    def __init__(self, evaluator, boundary, params_vary,
                 interface_energy,
                 num_density, init_grad_coeff=None):
        from cemc.phasefield import GradCoeffEvaluator

        if not isinstance(evaluator, GradCoeffEvaluator):
            raise TypeError("Evaluator has to be an instance of "
                            "GradCoeffEvaluator!")

        self.evaluator = evaluator
        self.boundary = boundary
        self.params_vary = params_vary
        self.grad_coeff = init_grad_coeff
        self.num_density = num_density
        self.interface_energy = interface_energy

        if self.grad_coeff is None:
            self.grad_coeff = np.ones(len(self.boundary))

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
            grid = np.linspace(b[free_param][0], b[free_param[1]], npoints)
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
            deriv = self.evaluator.deriv(variables, free_param)

            if np.any(free_energy < 0.0):
                raise RuntimeError("It appears like forming an interface "
                                   "lowers the energy!")

            integrand = np.sqrt(free_energy)
            grad_terms = np.zers_like(integrand) + self.grad_coeff[free_energy]
            for varying in varying_params:
                grad_terms += self.grad_coeff[varying]*deriv[varying, :]**2

            integral = np.trapz(integrand*np.sqrt(grad_terms))
            integrals[interface].append(self.num_density*integral)
        return integrals

    def solve(self):
        def func(grad_coeff):
            self.grad_coeff = grad_coeff
            integrals = self.calculate_integrals()
            rhs = []
            lhs = []
            for interface in integrals.keys():
                rhs.append(integrals[interface])
                lhs.append(self.interface_energy[interface])
            return np.array(lhs) - np.array(rhs)

        sol = newton(func, self.grad_coeff)
        self.grad_coeff = sol
        return self.grad_coeff
