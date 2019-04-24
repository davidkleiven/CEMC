import unittest
try:
    from cemc.phasefield import LinearInvolution, FractionalInvolution
    available = True
    reason = ""
except ImportError as exc:
    available = False
    reason = str(exc)


class TestInvolution(unittest.TestCase):
    def test_linear_is_involution(self):
        if not available:
            self.skipTest(reason)

        invol = LinearInvolution(xmax=2.0)
        x = 1.67
        y = invol(x)
        self.assertAlmostEqual(x, invol(y))

    def test_linear_deriv(self):
        if not available:
            self.skipTest(reason)

        invol = LinearInvolution(xmax=2.0)
        x = 1.23
        deriv = invol.deriv(x)

        dx = 1E-7
        fd_deriv = (invol(x+dx) - invol(x))/dx
        self.assertAlmostEqual(deriv, fd_deriv)

    def test_fractional_invol(self):
        if not available:
            self.skipTest(reason)

        invol = FractionalInvolution(xmax=2.0, k=7)
        x = 0.97
        y = invol(x)
        self.assertAlmostEqual(x, invol(y))

    def test_fractional_deriv(self):
        if not available:
            self.skipTest(reason)

        invol = FractionalInvolution(xmax=2.0, k=2)
        x = 1.87
        deriv = invol.deriv(x)

        dx = 1E-7
        fd_deriv = (invol(x+dx) - invol(x))/dx
        self.assertAlmostEqual(deriv, fd_deriv)

if __name__ == "__main__":
    from cemc import TimeLoggingTestRunner
    unittest.main(testRunner=TimeLoggingTestRunner)