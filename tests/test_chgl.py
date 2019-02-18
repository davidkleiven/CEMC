import unittest
import os
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

    def get_chgl(self):
        dim = 2
        L = 32
        num_gl_fields = dim
        M = 1.0
        alpha = 0.1
        dt = 0.001
        gl_damping = M
        grad_coeff = [[alpha, 0.5*alpha], [0.5*alpha, alpha]]
        return PyCHGL(dim, L, self.prefix, num_gl_fields, M, alpha, dt, gl_damping, 
                      grad_coeff)

    def test_run(self):
        if not available:
            self.skipTest(reason)
        chgl = self.get_chgl()
        chgl.random_initialization([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
        chgl.run(5, 1000)

    def tearDown(self):
        super(TestCHGL, self).tearDown()
        try:
            os.remove("chgl00000001000.grid")
            os.remove("chgl.grid")
        except IOError:
            pass


if __name__ == "__main__":
    from cemc import TimeLoggingTestRunner
    unittest.main(testRunner=TimeLoggingTestRunner)
    