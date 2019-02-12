import unittest
import numpy as np
import os
try:
    from phasefield_cxx import PyCahnHilliardPhaseField
    from phasefield_cxx import PyCahnHilliard
    available = True
    reason = ""
except ImportError as exc:
    available = False
    reason = str(exc)
    print(str(exc))


class TestCahnHilliardPhaseField(unittest.TestCase):
    def test_run_without_errors(self):
        if not available:
            self.skipTest(reason)
        coeff = [5.0, 4.0, 3.0, 2.0, 1.0]
        free = PyCahnHilliard(coeff)

        # Initialize a small 2D calculation
        L = 64
        M = 1.0
        dt = 0.01
        alpha = 1.0
        sim = PyCahnHilliardPhaseField(2, L, "cahnhill", free, M, dt, alpha)
        sim.run(100, 20)

    def tearDown(self):
        super(TestCahnHilliardPhaseField, self).tearDown()
        try:
            os.remove("cahnhill0.vti")
            os.remove("cahnhill20.vti")
            os.remove("cahnhill40.vti")
            os.remove("cahnhill60.vti")
            os.remove("cahnhill80.vti")
        except IOError:
            pass

if __name__ == "__main__":
    from cemc import TimeLoggingTestRunner
    unittest.main(testRunner=TimeLoggingTestRunner)
