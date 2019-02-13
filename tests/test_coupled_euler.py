import unittest
import numpy as np
try:
    from cemc.phasefield import CoupledEuler
    available = True
    reason = ""
except ImportError as exc:
    available = False
    reason = str(exc)


def rhs_harmonic_osc(y):
    return -y[1, :]


class TestCoupledEuler(unittest.TestCase):
    def test_linear_ode(self):
        if not available:
            self.skipTest(reason)
        x = np.linspace(0.0, np.pi, 100)
        b_vals = [[1.0, -1.0]]
        rhs = [rhs_harmonic_osc]
        euler = CoupledEuler(x, rhs, b_vals)
        expected = np.cos(x)
        sol = euler.solve()
        numerical = sol(x)[1, :]
        self.assertTrue(np.allclose(numerical, expected))


if __name__ == "__main__":
    from cemc import TimeLoggingTestRunner
    unittest.main(testRunner=TimeLoggingTestRunner)
