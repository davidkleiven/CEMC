import unittest
import os
import numpy as np
try:
    from phasefield_cxx import PyCHGL
    from phasefield_cxx import PyPolynomialTerm
    available = True
    reason = ""
except ImportError as exc:
    available = False
    reason = str(exc)


class TestCHGL(unittest.TestCase):
    prefix = "chgl"
    L = 32

    def get_chgl(self):
        dim = 2
        num_gl_fields = dim
        M = 1.0
        alpha = 0.1
        dt = 0.001
        gl_damping = M
        grad_coeff = [[alpha, 0.5*alpha], [0.5*alpha, alpha]]
        return PyCHGL(dim, self.L, self.prefix, num_gl_fields, M, alpha, dt, gl_damping, 
                      grad_coeff)

    def test_run(self):
        if not available:
            self.skipTest(reason)
        chgl = self.get_chgl()
        chgl.random_initialization([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
        chgl.run(5, 1000)

    def test_npy_array(self):
        if not available:
            self.skipTest(reason)

        chgl = self.get_chgl()
        array = [np.random.rand(self.L, self.L),
                 np.random.rand(self.L, self.L),
                 np.random.rand(self.L, self.L)]
        chgl.from_npy_array(array)
        from_chgl = chgl.to_npy_array()
        for arr1, arr2 in zip(array, from_chgl):
            self.assertTrue(np.allclose(arr1, arr2))

    def test_exceptions(self):
        if not available:
            self.skipTest()

        chgl = self.get_chgl()
        # Test wrong number of fields
        array = [np.random.rand(self.L, self.L),
                 np.random.rand(self.L, self.L)]
        with self.assertRaises(ValueError):
            chgl.from_npy_array(array)

        # Wrong dimension on one of the fields
        array = [np.random.rand(self.L, self.L),
                 np.random.rand(self.L, self.L),
                 np.random.rand(self.L, 2*self.L)]
        with self.assertRaises(ValueError):
            chgl.from_npy_array(array)

        # Wrong dimension on one field
        array = [np.random.rand(self.L, self.L),
                 np.random.rand(self.L, self.L),
                 np.random.rand(self.L, 2*self.L, 2)]
        with self.assertRaises(ValueError):
            chgl.from_npy_array(array)

    def tearDown(self):
        super(TestCHGL, self).tearDown()
        try:
            os.remove("chgl00000001000.grid")
            os.remove("chgl.grid")
        except OSError:
            pass


if __name__ == "__main__":
    from cemc import TimeLoggingTestRunner
    unittest.main(testRunner=TimeLoggingTestRunner)
    