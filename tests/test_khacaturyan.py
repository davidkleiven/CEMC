import unittest
import numpy as np

try:
    from cemc.tools import Khachaturyan
    available = True
    reason = ""
except ImportError as exc:
    available = False
    reason = str(exc)

class TestKhacaturyan(unittest.TestCase):
    K = 50.0
    G = 26.0

    def get_isotropic_tensor(self):
        C = np.zeros((6, 6))
        C[0, 0] = C[1, 1] = C[2, 2] = self.K + 4*self.G/3.0
        C[0, 1] = C[0, 2] = \
        C[1, 0] = C[1, 2] = \
        C[2, 0] = C[2, 1] = self.K - 2.0*self.G/3.0
        C[3, 3] = C[4, 4] = C[5, 5] = 2*self.G
        return C

    @property
    def poisson(self):
        return 0.5*(3*self.K - 2*self.G)/(3*self.K + self.G)

    def isotropic_green_function(self, k):
        return np.eye(3)/self.G - 0.5*np.outer(k, k)/(self.G*(1.0 - self.poisson))

    def test_isotropic(self):
        if not available:
            self.skipTest(reason)
        
        strain = Khachaturyan(elastic_tensor=self.get_isotropic_tensor())
        k = np.array([5.0, -2.0, 7.0])
        khat = k/np.sqrt(k.dot(k))
        zeroth = strain.zeroth_order_green_function(khat)
        self.assertTrue(np.allclose(zeroth, self.isotropic_green_function(khat)))


if __name__ == "__main__":
    from cemc import TimeLoggingTestRunner
    unittest.main(testRunner=TimeLoggingTestRunner)

