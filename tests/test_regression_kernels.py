import unittest
import numpy as np
try:
    from phasefield_cxx import PyQuadraticKernel
    from phasefield_cxx import PyKernelRegressor
    from phasefield_cxx import PyGaussianKernel
    from cemc.phasefield.phasefield_util import fit_kernel
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

    def test_kernel_regressor(self):
        if not available:
            self.skipTest(reason)

        width = 2.0
        kernel = PyQuadraticKernel(2.0)

        coeff = [0.0, 2.0, -3.0, 0.0]
        regressor = PyKernelRegressor(-12.0, 12.0)
        regressor.set_kernel(kernel)
        regressor.set_coeff(coeff)

        center1 = -4.0
        center2 = 4.0
        self.assertAlmostEqual(regressor.evaluate(center1),
                               coeff[1]*kernel.evaluate(0.0))
        self.assertAlmostEqual(regressor.evaluate(center2),
                               coeff[2]*kernel.evaluate(0.0))
        self.assertAlmostEqual(regressor.deriv(center1),
                               coeff[1]*kernel.deriv(0.0))
        self.assertAlmostEqual(regressor.deriv(center2),
                               coeff[2]*kernel.deriv(0.0))

        # Try outside domain
        self.assertAlmostEqual(regressor.evaluate(1000.0), 0.0)
        self.assertAlmostEqual(regressor.evaluate(-1000.0), 0.0)

    def test_fit_kernel(self):
        if not available:
            self.skipTest(reason)

        width = 4.1
        kernel = PyQuadraticKernel(width)
        x = np.linspace(0.0, 10.0, 500, endpoint=True)
        y = x**2
        num_kernels = 321
        regressor = fit_kernel(x=x, y=y, num_kernels=num_kernels, kernel=kernel)
        y_fit = regressor.evaluate(x)
        self.assertTrue(np.allclose(y, y_fit))

    def test_gaussian_kernel(self):
        if not available:
            self.skipTest(reason)

        width = 2.0
        kernel = PyGaussianKernel(width)

        # Confirm that the integral is 1
        w = np.linspace(-10*width, 10*width, 20000).tolist()

        values = [kernel.evaluate(x) for x in w]
        integral = np.trapz(values, dx=w[1] - w[0])
        self.assertAlmostEqual(integral, 1.0, places=4)

        # Check some values
        expected = np.exp(-0.5)*0.5/np.sqrt(2*np.pi)
        self.assertAlmostEqual(kernel.evaluate(width), expected)
        self.assertAlmostEqual(kernel.evaluate(-width), expected)

        # Check derivatives
        self.assertAlmostEqual(kernel.deriv(0.0), 0.0)
        self.assertAlmostEqual(kernel.deriv(width), expected/width)

if __name__ == "__main__":
    from cemc import TimeLoggingTestRunner
    unittest.main(testRunner=TimeLoggingTestRunner)