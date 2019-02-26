from scipy.optimize import newton


class HyperbolicTangentBVPSolver(object):
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
        return 0.5*height*np.tanh(x/w) + bv[0]

    def _basis_func_deriv(self, x, var_num):
        h = self._height(var_num)
        w = self.widths[var_num]
        return 0.5*h/(w*np.cosh(x/w)**2)

    def integrate_rhs(self):
        """Integrate the right hand side."""
        rhs_values = []
        profile = np.zeros((len(self.rhs, len(self.mesh_points))))
        for bf in profile.shape[0]:
            profile[bf, :] = self._basis_func(self.mesh_points, bf)

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
