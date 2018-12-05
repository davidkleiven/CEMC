"""Unittest for the Eshelby tensor."""
try:
    import unittest
    from cemc_cpp_code import PyEshelbyTensor, PyEshelbySphere
    from itertools import product
    import numpy as np
    from cemc.tools import rotate_rank4_mandel
    from cemc.tools import to_mandel
    from cemc.tools import rotate_tensor
    from cemc.tools import to_full_tensor, rotate_tensor
    from cemc.tools import to_mandel, to_full_tensor
    from cemc.tools import to_mandel_rank4, to_full_rank4
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

    def test_mandel_transformation(self):
        if not available:
            self.skipTest(reason)
        mandel_vec = np.linspace(1.0, 6.0, 6)
        mandel_full = to_full_tensor(mandel_vec)
        self.assertTrue(np.allclose(to_mandel(mandel_full), mandel_vec))

        # Try rank for tensors
        mandel_tensor = np.random.rand(6, 6)
        mandel_full = to_full_rank4(mandel_tensor)
        self.assertTrue(np.allclose(to_mandel_rank4(mandel_full), mandel_tensor))

    def test_tensor_rotation(self):
        if not available:
            self.skipTest(reason)
        tens = np.linspace(1.0, 6.0, 6)
        ca = np.cos(0.3)
        sa = np.sin(0.3)
        rot_matrix = np.array([[ca, sa, 0.0],
                               [-sa, ca, 0.0],
                               [0.0, 0.0, 1.0]])
        
        # Rotate the rank 2 tensor
        full_tensor = to_full_tensor(tens)
        rotated_tensor = rotate_tensor(full_tensor, rot_matrix)

        x = np.array([-0.1, 0.5, 0.9])

        x_rot = rot_matrix.dot(x)

        # If we contract all indices the scalar product should remain the same
        scalar1 = x.dot(full_tensor).dot(x)
        scalar2 = x_rot.dot(rotated_tensor).dot(x_rot)
        self.assertAlmostEqual(scalar1, scalar2)

    def test_rank4_tensor_rotation(self):
        if not available:
            self.skipTest(reason)
        
        vec = np.random.rand(6)
        full2x2 = to_full_tensor(vec)
        
        ca = np.cos(0.8)
        sa = np.sin(0.8)
        rot_matrix = np.array([[ca, sa, 0.0],
                               [-sa, ca, 0.0],
                               [0.0, 0.0, 1.0]])

        tensor = np.random.rand(6, 6)
        rotated = rotate_rank4_mandel(tensor, rot_matrix)

        rotated2x2 = rotate_tensor(full2x2, rot_matrix)
        rot_vec = to_mandel(rotated2x2)

        # Contract indices
        scalar1 = vec.dot(tensor).dot(vec)
        scalar2 = rot_vec.dot(rotated).dot(rot_vec)
        self.assertAlmostEqual(scalar1, scalar2)

    def test_rotation_isotropic_tensor(self):
        if not available:
            self.skipTest(reason)
        
        tensor = np.zeros((6, 6))

        # Construct an isotropic elasticity tensor
        K = 76
        mu = 11.0
        tensor[0, 0] = tensor[1, 1] = tensor[2, 2] = K + 4.0*mu/3.0
        tensor[3, 3] = tensor[4, 4] = tensor[5, 5] = 2*mu

        tensor[0, 1] = tensor[0, 2] = tensor[1, 0] = tensor[1, 2] = \
        tensor[2, 0] = tensor[2, 1] = K - 2.0*mu/3.0

        # Apply a sequence of rotations
        ca = np.cos(np.pi/4.0)
        sa = np.sin(np.pi/4.0)
        rot_matrix = np.array([[ca, sa, 0.0],
                               [-sa, ca, 0.0],
                               [0.0, 0.0, 1.0]])
        rotated = rotate_rank4_mandel(tensor, rot_matrix)
        np.set_printoptions(precision=2)
        self.assertTrue(np.allclose(tensor, rotated))



if __name__ == "__main__":
    from cemc import TimeLoggingTestRunner
    unittest.main(testRunner=TimeLoggingTestRunner)
