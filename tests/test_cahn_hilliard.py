import unittest
import numpy as np
try:
    from cemc.phasefield import CahnHilliard
    from phasefield_cxx import PyCahnHilliard
    available = True
    reason = ""
except ImportError as exc:
    available = False
    reason = str(exc)
    print(str(exc))


class TestCahnHilliard(unittest.TestCase):
    def test_fit(self):
        if not available:
            self.skipTest(reason)
        x = np.linspace(0.0, 1.0)
        a = 1.0
        b = 2.0
        c = 3.0
        d = 4.0
        e = 5.0
        G = a + b*x + c*x**2 + d*x**3 + e*x**4

        cahn = CahnHilliard(degree=4)
        cahn.fit(x, G)
        expected = [e, d, c, b, a]
        self.assertTrue(np.allclose(cahn.coeff, expected))

    def test_cython(self):
        if not available:
            self.skipTest(reason)
        coeff = [4.0, 3.0, 2.0, 1.0]
        cahn = CahnHilliard(degree=3, coeff=coeff)
        cython_cahn = PyCahnHilliard(coeff)

        x_values = [2.0, 3.0, -1.0]
        for x in x_values:
            self.assertAlmostEqual(cahn.evaluate(x), cython_cahn.evaluate(x))

    def test_raise_inconsistent_args(self):
        if not available:
            self.skipTest(reason)
        coeff = [4.0, 3.0, 2.0, 1.0]
        self.assertRaises(ValueError, CahnHilliard, coeff=coeff)

    def test_derivative(self):
        if not available:
            self.skipTest(reason)
        coeff = [4.0, 3.0, 2.0, 1.0]
        cahn = CahnHilliard(degree=3, coeff=coeff)
        cython_cahn = PyCahnHilliard(coeff)
        x_values = [2.0, 3.0, -1.0]
        for x in x_values:
            self.assertAlmostEqual(cahn.deriv(x), cython_cahn.deriv(x))

if __name__ == "__main__":
    from cemc import TimeLoggingTestRunner
    unittest.main(testRunner=TimeLoggingTestRunner)
