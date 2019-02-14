import unittest
try:
    from phasefield_cxx import PyPolynomialTerm
    reason = ""
    available = True
except ImportError as exc:
    available = False
    reason = str(exc)


class TestPolynomialTerm(unittest.TestCase):
    def test_poly_term(self):
        power = [2]
        term = PyPolynomialTerm(power, 1)
        self.assertAlmostEqual(term.evaluate([2.0]), 4.0)
        self.assertAlmostEqual(term.deriv([2.0], 0), 4.0)

        power = [2, 3]
        term = PyPolynomialTerm(power, 1)
        self.assertAlmostEqual(term.evaluate([2.0, -4.0]), -60.0)
        self.assertAlmostEqual(term.deriv([2.0, -4.0], 0), 4.0)
        self.assertAlmostEqual(term.deriv([2.0, -4.0], 1), 3*16)

        term = PyPolynomialTerm(power, 2)
        self.assertAlmostEqual(term.evaluate([2.0, -4.0]), 3600.0)
        self.assertAlmostEqual(term.deriv([2.0, -4.0], 0), 2*-60.0*2*2.0)
        self.assertAlmostEqual(term.deriv([2.0, -4.0], 1), 2*-60.0*3*16.0)

if __name__ == "__main__":
    from cemc import TimeLoggingTestRunner
    unittest.main(testRunner=TimeLoggingTestRunner)