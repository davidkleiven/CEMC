import unittest
from ase.units import kB
import numpy as np
avail_msg = ""
try:
    from cemc.mcmc.exponential_weighted_averager import ExponentialWeightedAverager
    available = True
except ImportError exc:
    avail_msg = str(exc)
    available = False


class TestExpAverager(unittest.TestCase):
    def test_sum(self):
        if not available:
            self.skipTest(avail_msg)
        E = -np.linspace(0.0, 10., 200)
        T = 200
        exp_avg_0 = ExponentialWeightedAverager(T, order=0)
        exp_avg_1 = ExponentialWeightedAverager(T, order=1)

        for i in range(len(E)):
            exp_avg_0.add(E[i])
            exp_avg_1.add(E[i])
        beta = 1.0 / (kB*T)
        exact_0 = np.sum(np.exp(-beta*(E-np.min(E))))/len(E)
        exact_1 = np.sum(E*np.exp(-beta*(E-np.min(E))))/len(E)

        self.assertAlmostEqual(exact_0, exp_avg_0.average, places=5)
        self.assertAlmostEqual(exact_1, exp_avg_1.average, places=5)

    def test_add(self):
        if not available:
            self.skipTest(avail_msg)
        T = 200
        beta = 1.0 / (kB * T)
        exp_avg_0 = ExponentialWeightedAverager(T, order=0)
        exp_avg_1 = ExponentialWeightedAverager(T, order=0)
        E1 = -np.linspace(0.0, 10.0, 20)
        E2 = -np.linspace(3.0, 12.0, 20)
        for i in range(len(E1)):
            exp_avg_0.add(E1[i])
            exp_avg_1.add(E2[i])
        new_avg = exp_avg_0 + exp_avg_1
        E1 = E1.tolist()
        E2 = E2.tolist()
        E = np.array(E1 + E2)
        E0 = np.min(E)
        exact = np.sum(np.exp(-beta * (E-E0)))/len(E)
        self.assertAlmostEqual(new_avg.average, exact, places=5)

        # Test the opposite addition operation
        new_avg = exp_avg_1 + exp_avg_0
        self.assertAlmostEqual(new_avg.average, exact, places=5)


if __name__ == "__main__":
    from cemc import TimeLoggingTestRunner
    unittest.main(testRunner=TimeLoggingTestRunner)
