import unittest
import numpy as np
import os
try:
    from phasefield_cxx import PyAdaptiveTimeStepLogger
    available = True
    reason = ""
except ImportError as exc:
    available = False
    reason = str(exc)


class TestAdaptiveTimeStepLogger(unittest.TestCase):
    def test_write_single(self):
        if not available:
            self.skipTest(reason)

        fname = "logger.csv"
        logger = PyAdaptiveTimeStepLogger(fname)
        logger.log(50, 0.5)
        logger.log(100, 0.7)
        logger.log(200, 0.9)

        # Read the file
        data = np.loadtxt(fname, delimiter=",")
        self.assertTrue(np.allclose([50, 100, 200], data[:, 0]))
        self.assertTrue(np.allclose([0.5, 0.7, 0.9], data[:, 1]))
        os.remove(fname)

    def test_append(self):
        if not available:
            self.skipTest(reason)

        fname = "logger_append.csv"
        logger = PyAdaptiveTimeStepLogger(fname)

        logger.log(20, 5.0)
        logger.log(30, 1.0)

        del logger

        logger2 = PyAdaptiveTimeStepLogger(fname)
        logger2.log(40, 6.0)
        logger2.log(60, 30)

        data = np.loadtxt(fname, delimiter=",")
        self.assertTrue(np.allclose([20, 30, 40, 60], data[:, 0]))
        self.assertTrue(np.allclose([5.0, 1.0, 6.0, 30], data[:, 1]))
        os.remove(fname)

    def test_get_last(self):
        if not available:
            self.skipTest(reason)

        fname = "logger_get_last.csv"

        logger = PyAdaptiveTimeStepLogger(fname)
        logger.log(10, 0.3)
        logger.log(20, 4.0)
        logger.log(30, 0.1)

        last = logger.getLast()
        self.assertAlmostEqual(last["time"], 0.1)
        self.assertEqual(last["iter"], 30)
        os.remove(fname)


if __name__ == "__main__":
    from cemc import TimeLoggingTestRunner
    unittest.main(testRunner=TimeLoggingTestRunner)