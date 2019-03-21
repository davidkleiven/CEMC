import unittest
import numpy as np
try:
    from cemc.phasefield import GradientCoefficient
    from cemc.phasefield import GradientCoefficientRhsBuilder
    from scipy.interpolate import interp1d
    available = True
    reason = ""

    class SingleParameterRhs(GradientCoefficientRhsBuilder):
        def __init__(self, boundary_values):
            GradientCoefficientRhsBuilder.__init__(self, boundary_values)

        def grad(self, c):
            """Gradient with respect to all parameters."""
            return 4*c**3 - 4*c

        def evaluate(self, c):
            return (1.0 - c**2)**2

except ImportError as exc:
    available = False
    reason = str(exc)


class TestGradientCoeff(unittest.TestCase):
    def test_single_parameter(self):
        if not available:
            self.skipTest(reason)
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

        df_func = {
            (0, 1): df
        }
        rhs = SingleParameterRhs(boundary_values)
        solvers = ["collocation", "hypertangent"]
        for solver in solvers:
            grad_coeff = GradientCoefficient(rhs, mesh_points, density,
                                             interface_energy,
                                             params_vary, tol=1E-8, width=0.1,
                                             max_nodes=1000000, solver=solver)

            res = grad_coeff.find_gradient_coefficients()
            K = (3*interface_energy[(0, 1)]/8.0)**2
            msg = "{} failed".format(solver)
            self.assertAlmostEqual(res[0], K, places=7, msg=msg)

if __name__ == "__main__":
    from cemc import TimeLoggingTestRunner
    unittest.main(testRunner=TimeLoggingTestRunner)
