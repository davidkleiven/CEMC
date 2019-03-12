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
        if not available:
            self.skipTest(reason)
        power = [2]
        term = PyPolynomialTerm(power)
        self.assertAlmostEqual(term.evaluate([2.0]), 4.0)
        self.assertAlmostEqual(term.deriv([2.0], 0), 4.0)

        power = [2, 3]
        term = PyPolynomialTerm(power)
        self.assertAlmostEqual(term.evaluate([2.0, -4.0]), -256.0)
        self.assertAlmostEqual(term.deriv([2.0, -4.0], 0), -256.0)
        self.assertAlmostEqual(term.deriv([2.0, -4.0], 1), 3*4*16)

if __name__ == "__main__":
    from cemc import TimeLoggingTestRunner
    unittest.main(testRunner=TimeLoggingTestRunner)