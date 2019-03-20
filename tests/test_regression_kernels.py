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

        width = 0.5
        kernel = PyGaussianKernel(width)
        x = np.linspace(0.0, 10.0, 500, endpoint=True)
        y = x**2
        num_kernels = 60
        regressor = fit_kernel(x=x, y=y, num_kernels=num_kernels, kernel=kernel)
        y_fit = regressor.evaluate(x)
        self.assertTrue(np.allclose(y, y_fit, atol=1E-4))

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

    def test_to_dict_quadratic(self):
        if not available:
            self.skipTest(reason)
        width = 2.0
        kernel = PyQuadraticKernel(width)
        dict_repr = kernel.to_dict()
        self.assertAlmostEqual(width, dict_repr["width"])
        self.assertAlmostEqual(width, dict_repr["upper_limit"])
        self.assertAlmostEqual(-width, dict_repr["lower_limit"])

    def test_to_dict_gaussian(self):
        if not available:
            self.skipTest(reason)
        width = 2.0
        kernel = PyGaussianKernel(width)
        dict_repr = kernel.to_dict()
        self.assertAlmostEqual(width, dict_repr["std_dev"])
        self.assertAlmostEqual(5*width, dict_repr["upper_limit"])
        self.assertAlmostEqual(-5*width, dict_repr["lower_limit"])

    def test_to_dict(self):
        if not available:
            self.skipTest(reason)

        width = 2.0
        kernel = PyGaussianKernel(width)
        regressor = PyKernelRegressor(0.0, 1.0)
        coeff = np.linspace(0.0, 10.0, 100)
        regressor.set_coeff(coeff)
        regressor.set_kernel(kernel)

        dict_repr = regressor.to_dict()
        self.assertAlmostEqual(0.0, dict_repr["xmin"])
        self.assertAlmostEqual(1.0, dict_repr["xmax"])
        self.assertEqual("gaussian", dict_repr["kernel_name"])
        self.assertTrue(np.allclose(coeff, dict_repr["coeff"]))

    def test_quadratic_from_dict(self):
        if not available:
            self.skipTest(reason)
        width = 2.0
        kernel = PyQuadraticKernel(width)
        dict_repr = kernel.to_dict()

        x0 = -0.34
        k0 = kernel.evaluate(x0)
        kernel.from_dict(dict_repr)
        self.assertAlmostEqual(k0, kernel.evaluate(x0))

    def test_gaussian_from_dict(self):
        if not available:
            self.skipTest(reason)
        width = 2.0
        kernel = PyGaussianKernel(width)
        dict_repr = kernel.to_dict()

        x0 = -0.34
        k0 = kernel.evaluate(x0)
        kernel.from_dict(dict_repr)
        self.assertAlmostEqual(k0, kernel.evaluate(x0))

    def test_from_dict(self):
        if not available:
            self.skipTest(reason)
        width = 2.0
        kernel = PyGaussianKernel(width)
        regressor = PyKernelRegressor(0.0, 1.0)
        coeff = np.linspace(0.0, 10.0, 100)
        regressor.set_coeff(coeff)
        regressor.set_kernel(kernel)

        # Evaluate at points
        x_values = [0.0, 2.0, -2.0, 5.0]
        y_values_orig = regressor.evaluate(x_values)

        dict_repr = regressor.to_dict()

        regressor.from_dict(dict_repr)
        y_values_new = regressor.evaluate(x_values)
        self.assertTrue(np.allclose(y_values_orig, y_values_new))

        # Verify that exception is raised if a wrong kernel is passed
        dict_repr["kernel_name"] = "quadratic"
        with self.assertRaises(ValueError):
            regressor.from_dict(dict_repr)

if __name__ == "__main__":
    from cemc import TimeLoggingTestRunner
    unittest.main(testRunner=TimeLoggingTestRunner)