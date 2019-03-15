import unittest
import numpy as np
try:
    from phasefield_cxx import PyQuadraticKernel
    available = True
    reason = ""
except ImportError as exc:
    available = False
    reason = str(exc)
print(reason)


class TestRegressionKernels(unittest.TestCase):
    def test_quadtratic_kernel(self):
        if not available:
            self.skipTest(reason)

        width = 2.0
        kernel = PyQuadraticKernel(2.0)

        # Confirm that the integral is 1
        w = np.linspace(-width, width, 20000).tolist()

        values = [kernel.evaluate(x) for x in w]
        integral = np.trapz(values, dx=w[1] - w[0])
        self.assertAlmostEqual(integral, 1.0, places=4)

        # Check some values
        self.assertAlmostEqual(kernel.evaluate(width), 0.0)
        self.assertAlmostEqual(kernel.evaluate(-width), 0.0)

        # Check an intermediate value
        x = 0.23
        expect = 0.75*(1.0 - (x/width)**2)/(width)
        self.assertAlmostEqual(kernel.evaluate(x), expect)


if __name__ == "__main__":
    from cemc import TimeLoggingTestRunner
    unittest.main(testRunner=TimeLoggingTestRunner)