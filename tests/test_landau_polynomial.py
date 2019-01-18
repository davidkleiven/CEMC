import unittest
import numpy as np
try:
    from cemc.tools import TwoPhaseLandauPolynomial
    from cemc.tools.landau_polynomial import MultivariatePolynomial
    available = True
    reason = ""
except ImportError as exc:
    available = False
    reason = str(exc)
    print(str(exc))

class TestLandauPolynomial(unittest.TestCase):
    def test_num_coeff(self):
        if not available:
            self.skipTest(reason)
        poly = TwoPhaseLandauPolynomial(strain_order=6)
        num_coeff = poly.num_strain_coeff
        self.assertEqual(num_coeff, 5)

    def test_fit(self):
        conc1 = np.linspace(0.0, 0.3, 100)
        F1 = conc1**2
        conc2 = np.linspace(0.6, 1.0, 100)
        F2 = 0.5*(conc2-0.9)**2
        poly = TwoPhaseLandauPolynomial(conc_order=2, strain_order=14,
                                        c1=0.0, c2=0.5)
        fname = "two_phase_landau_fit.csv"
        poly.fit(conc1, F1, conc2, F2, fname=fname)
        self.assertAlmostEqual(poly.polynomials["phase1"].coeff[0], 1.0)

    def test_multivariate_polynomial(self):
        coeff = [1.0, 2.0, 3.0]
        powers = [(1, 2, 3), (0, 2, 0), (0, 0, 3)]
        multipol = MultivariatePolynomial(coeff=coeff, powers=powers)
        x = [1.0, 2.0, 3.0]
        self.assertAlmostEqual(multipol(x), 7*27 + 8)

        self.assertAlmostEqual(multipol.deriv(x, dim=1), 4*27 + 8)

        multipol = MultivariatePolynomial(coeff=coeff, powers=powers, cyclic=True)
        self.assertAlmostEqual(multipol(x), 7*27 + 28 + 21 + 48)




if __name__ == "__main__":
    from cemc import TimeLoggingTestRunner
    unittest.main(testRunner=TimeLoggingTestRunner)