import unittest
import numpy as np
try:
    from cemc.phasefield import HyperbolicTangentBVPSolver
    available = True
    reason = ""
except ImportError as exc:
    available = False
    reason = str(exc)


class TestHyperTangentBVP(unittest.TestCase):
    def test_excact(self):
        boundary = [[0.0, 1.0]]

        K = 1.0
        delta = 1.0
        w = 1.23

        def rhs(y):
            x = 2*y[1, :]/delta
            factor1 = 2*K/w**2
            factor2 = 1 - x
            factor3 = (1.0 - (x - 1)**2)
            return factor1*factor2*factor3

        x = np.linspace(-5.0, 5.0, 100)
        solver = HyperbolicTangentBVPSolver([rhs], x, boundary,
                                            mass_terms=[2*K])
        solver.solve()
        res = solver.widths[0]
        self.assertAlmostEqual(res, w, places=2)

    def test_exact_two_parameter(self):
        x = np.linspace(-5.0, 5.0, 100)
        boundary = [[0.0, 1.0], [0.0, 1.5]]
        K1 = 1.0
        K2 = 2.0
        delta1 = 1.0
        delta2 = 1.5
        w = 1.43

        def rhs1(y):
            x1 = 2*y[1, :]/delta1
            x2 = 2*y[3, :]/delta2

            factor11 = -2*K1*delta1/w**2
            factor12 = (x2 - 1)
            factor13 = 1 - (x1-1)**2
            return factor11*factor12*factor13

        def rhs2(y):
            x1 = 2*y[1, :]/delta1
            x2 = 2*y[3, :]/delta2
            factor21 = -2*K2*delta2/w**2
            factor22 = (x1 - 1)
            factor23 = (1 - (x2-1)**2)
            return factor21*factor22*factor23

        solver = HyperbolicTangentBVPSolver([rhs1, rhs2], x, boundary, 
                                            mass_terms=[2*K1, 2*K2])
        solver.solve()
        res = solver.widths
        self.assertTrue(np.allclose(res, [w, w], atol=1E-2))

if __name__ == "__main__":
    from cemc import TimeLoggingTestRunner
    unittest.main(testRunner=TimeLoggingTestRunner)