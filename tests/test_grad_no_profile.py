import unittest
import numpy as np

try:
    from cemc.phasefield import GradientCoeffNoExplicitProfile
    from cemc.phasefield import GradCoeffEvaluator
    available = True
    reason = ""

    class TwoParameterEvaluator(GradCoeffEvaluator):

        # NOTE: Do not change these numbers!
        A = 1.0
        B = 8.0

        def evaluate(self, x, free_param):
            if free_param == 0:
                # First varies, in this model
                # the second field is then given
                return self.A*x[0]**2 + self.B*x[0]**2
            else:
                return self.B*x[1]**2

        def deriv(self, x, free_param):
            result = np.zeros((len(x), len(x[free_param])))
            if free_param == 0:
                result[0, :] = 1.0
                result[1, :] = 1.0
            else:
                result[1, :] = 1.0
            return result

except ImportError as exc:
    available = False
    reason = str(exc)


class TestGradCoeffNoProfile(unittest.TestCase):
    def test_two_parameter_model(self):
        if not available:
            self.skipTest(reason)

        evaluator = TwoParameterEvaluator()
        interface_energy = {
            (0, 1): 5.0,
            (0, 2): 2.0
        }

        boundary = {
            (0, 1): [[0.0, 1.0], [0.0, 1.0]],
            (0, 2): [[1.0, 1.0], [0.0, 1.0]]
        }

        params_vary = {
            (0, 1): [0, 1],
            (0, 2): [1]
        }

        n_density = 1.0
        grad = GradientCoeffNoExplicitProfile(
            evaluator, boundary, interface_energy, params_vary,
            n_density, init_grad_coeff=np.array([0.3, 2.0]))
        coeff = grad.solve()

        Kn = (1.0/evaluator.B)*(interface_energy[(0, 2)]/n_density)**2
        Kc = (1.0/(evaluator.A + evaluator.B)) * \
            (interface_energy[(0, 1)]/n_density)**2
        Kc -= Kn

        self.assertAlmostEqual(Kc, coeff[0])
        self.assertAlmostEqual(Kn, coeff[1])


if __name__ == "__main__":
    from cemc import TimeLoggingTestRunner
    unittest.main(testRunner=TimeLoggingTestRunner)