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

    def test_fit(self):
        if not available:
            self.skipTest(reason)
        error = False
        msg = ""
        try:
            conc1 = np.linspace(0.0, 0.3, 100)
            F1 = conc1**2
            conc2 = np.linspace(0.6, 1.0, 100)
            F2 = 0.5*(conc2-0.9)**2
            poly = TwoPhaseLandauPolynomial(c1=0.0, c2=0.5)
            poly.fit(conc1, F1, conc2, F2)
        except Exception as exc:
            error = True
            msg = str(exc)

        self.assertFalse(error, msg=msg)




if __name__ == "__main__":
    from cemc import TimeLoggingTestRunner
    unittest.main(testRunner=TimeLoggingTestRunner)