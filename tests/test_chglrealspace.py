import unittest
import os
import numpy as np
try:
    from phasefield_cxx import PyCHGLRealSpace
    from phasefield_cxx import PyTwoPhaseLandau
    from phasefield_cxx import PyPolynomial
    from phasefield_cxx import PyKernelRegressor
    from phasefield_cxx import PyGaussianKernel
    available = True
    reason = ""
except ImportError as exc:
    available = False
    reason = str(exc)
    print(reason)

class TestCHGLRealSpace(unittest.TestCase):
    prefix = "chglrealspace"
    L = 32

    def get_chgl(self):
        dim = 2
        num_gl_fields = dim
        M = 1.0
        alpha = 0.1
        dt = 0.001
        gl_damping = M
        grad_coeff = [[alpha, 0.5*alpha], [0.5*alpha, alpha]]
        return PyCHGLRealSpace(dim, self.L, self.prefix, num_gl_fields, M, alpha, dt, gl_damping, 
                      grad_coeff)

    def test_chglrealspace(self):
        if not available:
            self.skipTest(reason)

        chgl = self.get_chgl()
        chgl.build2D()

if __name__ == "__main__":
    from cemc import TimeLoggingTestRunner
    unittest.main(testRunner=TimeLoggingTestRunner)