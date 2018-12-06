import unittest
import traceback
try:
    from cemc.tools import IsotropicStrainEnergy
    available = True
    skip_reason = ""
except ImportError as exc:
    skip_reason = str(exc)
    available = False


class TestIsotropicStrain(unittest.TestCase):
    def test_no_throw(self):
        if not available:
            self.skipTest(skip_reason)

        no_throw = True
        msg = ""
        try:
            iso = IsotropicStrainEnergy(bulk_mod=76, shear_mod=26)

            iso.plot(princ_misfit=[-0.1, 0.1, 0.1], show=False, theta=3.14/2.0)
        except Exception as exc:
            msg = type(exc).__name__ + ": " + str(exc)
            msg += "\n" + traceback.format_exc()
            no_throw = False
        
        self.assertTrue(no_throw, msg=msg)

if __name__ == "__main__":
    from cemc import TimeLoggingTestRunner
    unittest.main(testRunner=TimeLoggingTestRunner)