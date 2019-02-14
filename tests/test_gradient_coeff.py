import unittest
import numpy as np
try:
    from cemc.phasefield import GradientCoefficient
    from scipy.interpolate import interp1d
    available = True
    reason = ""
except ImportError as exc:
    available = False
    reason = str(exc)


class TestGradientCoeff(unittest.TestCase):
    def test_single_parameter(self):
        c = np.linspace(-1.0, 1.0, 100)
        f = (1.0 - c**2)**2

        # Construct the partial derivatives with respect to the variable
        rhs = interp1d(c, 4*c**3 - 4.0*c, bounds_error=None,
                       fill_value="extrapolate")
        df = interp1d(c, f, fill_value="extrapolate")
        density = 1.0
        interface_energy = {(0, 1): 2.0}
        boundary_values = {(0, 1): [[-1.0, 1.0]]}
        params_vary = {(0, 1): [0]}

        mesh_points = np.linspace(-4.0, 4.0, 100)

        def rhs_func(y):
            return rhs(y[1, :])

        grad_coeff = GradientCoefficient([rhs_func], mesh_points, density,
                                         interface_energy, df, boundary_values,
                                         params_vary, tol=1E-8, width=0.1,
                                         max_nodes=1000000)

        res = grad_coeff.find_gradient_coefficients()
        K = (3*interface_energy[(0, 1)]/8.0)**2
        self.assertAlmostEqual(res[0], K, places=2)

if __name__ == "__main__":
    from cemc import TimeLoggingTestRunner
    unittest.main(testRunner=TimeLoggingTestRunner)
