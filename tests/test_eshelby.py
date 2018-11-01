"""Unittest for the Eshelby tensor."""
try:
    import unittest
    from cemc_cpp_code import PyEshelbyTensor, PyEshelbySphere
    from itertools import product
    import numpy as np
    available = True
    reason = ""
except ImportError as exc:
    reason = str(exc)
    available = False


class TestEshelby(unittest.TestCase):
    """Class for testing the Eshelby tensor."""

    def test_minor_symmetry_tensor(self):
        """Test that the Eshelby tensor exhibt minor symmetry."""
        if not available:
            self.skipTest(reason)
        e_tensor = PyEshelbyTensor(6.0, 5.0, 4.0, 0.3)

        for indx in product([0, 1, 2], repeat=4):
            val1 = e_tensor(indx[0], indx[1], indx[2], indx[3])

            val2 = e_tensor(indx[0], indx[1], indx[3], indx[2])
            self.assertAlmostEqual(val1, val2)

            val2 = e_tensor(indx[1], indx[0], indx[3], indx[2])
            self.assertAlmostEqual(val1, val2)

            val2 = e_tensor(indx[1], indx[0], indx[2], indx[3])
            self.assertAlmostEqual(val1, val2)

    def test_mmtensor_ellipsoid(self):
        """Test constency of the Eshelby sphere class."""
        if not available:
            self.skipTest(reason)
        E = PyEshelbyTensor(3.0, 1.2, 0.25, 0.27)

        mmtensor = np.loadtxt("tests/test_data/mmtensor_ellipse.txt")
        mandel = np.array(E.aslist())
        self.assertTrue(np.allclose(mmtensor, mandel, atol=1E-5))

    def test_mmtensor_sphere(self):
        """Test consistency with MM Tensor for spherical inclusion."""
        if not available:
            self.skipTest(reason)
        E = PyEshelbySphere(6.0, 6.0, 6.0, 0.27)
        mandel = E.aslist()
        mmtensor = np.loadtxt("tests/test_data/mmtensor_sphere.txt")
        mandel = np.array(mandel)
        self.assertTrue(np.allclose(mmtensor, mandel, atol=1E-5))

    def test_mmtensor_oblate(self):
        """Test consistency with MM Tensor for oblate sphere."""
        if not available:
            self.skipTest(reason)
        E = PyEshelbyTensor(6.0, 6.0, 2.0, 0.27)
        mandel = E.aslist()
        mandel = np.array(mandel)
        mmtensor = np.loadtxt("tests/test_data/mmtensor_oblate.txt")
        self.assertTrue(np.allclose(mmtensor, mandel, atol=1E-5))

    def test_mmtensor_prolate(self):
        """Test consistency with MM Tensor for oblate sphere."""
        if not available:
            self.skipTest(reason)
        E = PyEshelbyTensor(6.0, 4.0, 4.0, 0.27)
        mandel = E.aslist()
        mandel = np.array(mandel)
        mmtensor = np.loadtxt("tests/test_data/mmtensor_prolate.txt")
        self.assertTrue(np.allclose(mmtensor, mandel, atol=1E-5))

if __name__ == "__main__":
    from cemc import TimeLoggingTestRunner
    unittest.main(testRunner=TimeLoggingTestRunner)
